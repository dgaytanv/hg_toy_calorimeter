"""Run python/bin/create_dataset.py <ini> <section> unchanged, writing ra_pickles instead of .djctd.

Runtime patches, nothing in python/ is edited:
1. DeepJetCore and HGCalML's datastructures are stubbed. dataset_creator imports them at module level, but uses
   them only to write .djctd (TrainData_NanoML, SimpleArray) and for compute_spectators_dist (find_pcas, off).
2. dataset_creator.get_djc_data returns the EventGenerator result dict itself, plus the t_idx its caller checks.
3. EventGenerator._gather_event_data also attaches the merged particle table as result['particles']. Row k of
   every column is the particle with truth_assignment == k. Derived columns: particles_n_hits,
   particles_truth_assisted_energy, particles_momentum_eta, particles_momentum_phi.
4. DatasetCreator.write_out writes the result dicts to ra_pickles, one file per num_events_per_djc_file events.
5. Optional ini key merge_particles_with_tracks (create_dataset.py ignores it) is passed to every EventGenerator.
   HEAD's default False never merges charged particles; the first event builder (91f4c1e) merged all.

Units are the EventGenerator's: result['rechit_*'] / result['truth_assignment_{x,y,z,vertex_*}'] in cm,
result['particles'] positions in mm (Geant4), energies in GeV.
"""
import configparser
import os
import runpy
import sys
import types

CREATE_DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python', 'bin', 'create_dataset.py')

# Columns present after EventGenerator._merge_particles whether or not a merge happened in the event.
PARTICLE_KEYS = ['particles_vertex_position_x', 'particles_vertex_position_y', 'particles_vertex_position_z',
                 'particles_pdgid', 'particles_momentum_direction_x', 'particles_momentum_direction_y',
                 'particles_momentum_direction_z', 'particles_kinetic_energy', 'particles_total_energy_deposited_all',
                 'particles_only_minbias', 'merged', 'particles_shower_class', 'particles_track_momentum',
                 'particles_first_active_impact_position_x', 'particles_first_active_impact_position_y',
                 'particles_first_active_impact_position_z']

for name, attrs in (('DeepJetCore', {'SimpleArray': None}),
                    ('datastructures', {}),
                    ('datastructures.TrainData_NanoML', {'TrainData_NanoML': None, 'find_pcas': None})):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module

import numpy as np
import calo.dataset_creator as dataset_creator
import calo.event_generator as event_generator
from ra_pickles import RandomAccessPicklesWriter


def result_as_event(result, compute_spectators_dist=True):
    return result, {'t_idx': result['truth_assignment'][..., np.newaxis]}


_gather_event_data = event_generator.EventGenerator._gather_event_data


def gather_event_data_with_particles(self, reduced_simulation):
    result = _gather_event_data(self, reduced_simulation)
    n = len(reduced_simulation['particles_kinetic_energy'])
    particles = {k: np.asarray(reduced_simulation[k]) for k in PARTICLE_KEYS}
    bad = [k for k, v in particles.items() if len(v) != n]
    if bad:
        raise RuntimeError('particle columns with the wrong length: %s' % bad)
    ta = np.asarray(result['truth_assignment'])
    if len(ta) and ta.max() >= n:
        raise RuntimeError('truth_assignment points past the particle table')
    # Derived columns: hits and truth-assisted energy (calibrated rechit energy of the particle's hits above threshold),
    # direction of the momentum at the vertex.
    hit = ta >= 0
    particles['particles_n_hits'] = np.bincount(ta[hit], minlength=n)
    particles['particles_truth_assisted_energy'] = np.bincount(ta[hit], weights=result['rechit_energy'][hit], minlength=n)
    mx, my, mz = (particles['particles_momentum_direction_%s' % c].astype(np.float64) for c in 'xyz')
    with np.errstate(divide='ignore', invalid='ignore'):
        particles['particles_momentum_eta'] = np.arcsinh(mz / np.hypot(mx, my))
    particles['particles_momentum_phi'] = np.arctan2(my, mx)
    result['particles'] = particles
    return result


def write_out(self, data):
    if data.qsize() == 0:
        return
    events = []
    while data.qsize() > 0:
        result, _ = data.get()
        events.append(result)
    writer = RandomAccessPicklesWriter(len(events), self.output_path)
    for event in events:
        writer.add(event)
    writer.close()
    print('build_events: wrote %d events to %s' % (len(events), self.output_path), flush=True)


dataset_creator.get_djc_data = result_as_event
event_generator.EventGenerator._gather_event_data = gather_event_data_with_particles
dataset_creator.DatasetCreator.write_out = write_out

config = configparser.ConfigParser()
config.read(sys.argv[1])
if 'merge_particles_with_tracks' in config[sys.argv[2]]:
    merge_with_tracks = config[sys.argv[2]].getboolean('merge_particles_with_tracks')
    _init = event_generator.EventGenerator.__init__

    def init_with_merging(self, *args, **kwargs):
        _init(self, *args, merge_particles_with_tracks=merge_with_tracks, **kwargs)

    event_generator.EventGenerator.__init__ = init_with_merging
    print('build_events: merge_particles_with_tracks =', merge_with_tracks, flush=True)

# Optional ini key draw_log=<file>: one line per loaded event with the particles_folder indices drawn for it
# (events dropped later for lack of truth are included), so two sets drawn from one pool can be compared.
if 'draw_log' in config[sys.argv[2]]:
    draw_log = config[sys.argv[2]]['draw_log']
    _load_n = dataset_creator.DatasetCreator.load_n

    def load_n_logged(self, reader, samples):
        data = _load_n(self, reader, samples)
        if reader is self.particles_iterator and len(samples):
            with open(draw_log, 'a') as f:
                f.write(' '.join(str(int(s)) for s in samples) + '\n')
        return data

    dataset_creator.DatasetCreator.load_n = load_n_logged
    print('build_events: logging particle draws to', draw_log, flush=True)

sys.argv = [CREATE_DATASET] + sys.argv[1:3]
runpy.run_path(CREATE_DATASET, run_name='__main__')
