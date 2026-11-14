"""build_events.py with hit times: every event also gets result['rechit_time'] (ns; -99 = no valid time) and
result['rechit_time_charge'] (the cell's in-time charge in fC), built by hit_time.py from the deposit times that the
generator of this branch stores (step_* keys of every simulation).

usage: build_events_time.py <ini> <section>     (from python/bin, with python/libs on PYTHONPATH; see the README)

It patches EventGenerator before running build_events.py unchanged (runpy), so that script's own patches wrap
these. Nothing in hg_toy_calorimeter is edited. The deposits that enter an event are collected at the two places where
the builder collects the energies:
- EventGenerator._gather_rechit_energy(simulations) without particle hit data (no phase cut: the particle set of the
  event): every deposit of those simulations;
- EventGenerator._filter_simulations(simulations, phase_cut) with a phase cut (pileup): only the deposits of the
  particles the cut keeps. The builder draws a scalar cut's window inside this method; here the window is drawn first,
  with the same draw, and passed on as a tuple, so that the particles kept are known.
The particle set of an event is one collision (one time and one beam-line position); every pileup simulation is its
own collision.

Out-of-time pileup (ini key time_oot_bunches: "cms" = the CMS Phase-2 scenario's crossings -3..+3, or a list such as
"-2,-1,1"; empty or absent: none): the builder is given
num_pu = (pileup per bunch crossing) x (1 + number of out-of-time bunches) simulations per event. The first share
is the in-time pileup and goes through the builder unchanged (truth, energies). The rest are split evenly over the
out-of-time bunches: they get no truth; under the same phase cut and window as the in-time pileup, their deposits,
shifted by 25 ns per bunch, add pulse-weighted energy to the in-time hits (hit_time.leak_energy, added to the raw
energies before the builder's noise cut) and, where they fall in the in-time bucket, enter the time of arrival.

Optional ini keys of the section (defaults in hit_time.DEFAULTS): time_threshold_fc, time_collision_sigma_ns,
time_collision_sigma_z_cm, time_noise_gev (the cell noise that sets the jitter; default = the builder's noise_sigma
key if present), time_jitter, time_oot_bunches.
"""
import configparser
import os
import runpy
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_EVENTS = os.path.join(HERE, 'build_events.py')
sys.path.insert(0, HERE)

for name, attrs in (('DeepJetCore', {'SimpleArray': None}),
                    ('datastructures', {}),
                    ('datastructures.TrainData_NanoML', {'TrainData_NanoML': None, 'find_pcas': None})):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module

import numpy as np  # noqa: E402
import calo.event_generator as event_generator  # noqa: E402
from calo import detector_math as dm  # noqa: E402

import hit_time  # noqa: E402

config = configparser.ConfigParser()
config.read(sys.argv[1])
section = config[sys.argv[2]]
PARAMS = dict(hit_time.DEFAULTS)
PARAMS['threshold_fc'] = section.getfloat('time_threshold_fc', PARAMS['threshold_fc'])
PARAMS['collision_sigma_ns'] = section.getfloat('time_collision_sigma_ns', PARAMS['collision_sigma_ns'])
PARAMS['noise_gev'] = section.getfloat('time_noise_gev', section.getfloat('noise_sigma', PARAMS['noise_gev']))
PARAMS['collision_sigma_z_cm'] = section.getfloat('time_collision_sigma_z_cm', PARAMS['collision_sigma_z_cm'])
PARAMS['jitter'] = section.getboolean('time_jitter', PARAMS['jitter'])
_oot = section.get('time_oot_bunches', '').replace(' ', '')
OOT = list(hit_time.CMS_OOT_BUNCHES) if _oot == 'cms' else [int(b) for b in _oot.split(',') if b]
if 0 in OOT or len(set(OOT)) != len(OOT):
    raise ValueError('time_oot_bunches must be distinct non-zero bunch crossings, got %s' % OOT)
PARAMS['oot_bunches'] = OOT
print('build_events_time: %s' % PARAMS, flush=True)

EG = event_generator.EventGenerator
_reset, _add, _gather, _filter, _event_data = EG.reset, EG.add, EG._gather_rechit_energy, EG._filter_simulations, EG._gather_event_data


def reset(self):
    _reset(self)
    self._time_store = []      # deposits of the in-time collisions
    self._time_oot = []        # deposits of the out-of-time collisions (shifted by 25 ns per bunch)
    self._time_minbias = False


def capture(store, simulations, keeps, minbias, bunch=0):
    hit_time.capture_steps(store, simulations, keeps, minbias, PARAMS['collision_sigma_ns'], np.random,
                           collision_sigma_z_cm=PARAMS['collision_sigma_z_cm'], bunch=bunch)


def draw_window(phase_cut):
    """The builder's own draw of a scalar phase cut's window, made here so that the window is known."""
    if phase_cut is None or isinstance(phase_cut, tuple):
        return phase_cut
    start = np.random.uniform(0, 2 * np.pi)
    return (start, np.fmod(start + phase_cut, 2 * np.pi))


def phase_keeps(simulations, window):
    keeps = []
    for sim in simulations:
        x = np.array(sim['particles_first_active_impact_position_x'])
        if len(x) == 0:
            keeps.append(np.zeros(0, bool))
            continue
        _eta, phi, _theta = dm.x_y_z_to_eta_phi_theta(x, np.array(sim['particles_first_active_impact_position_y']),
                                                      np.array(sim['particles_first_active_impact_position_z']))
        keeps.append(hit_time.kept_particles(np.asarray(phi), window))
    return keeps


def add(self, simulations, phase_cut=None, eta_cut=None, minbias=False):
    self._time_minbias = bool(minbias)
    if minbias and OOT:
        if eta_cut is not None:
            raise NotImplementedError('out-of-time pileup with an eta cut')
        n_in = len(simulations) // (1 + len(OOT))
        if n_in == 0:
            raise ValueError('%d pileup simulations for 1 + %d bunch crossings' % (len(simulations), len(OOT)))
        phase_cut = draw_window(phase_cut)               # one window for the in-time and the out-of-time pileup
        rest = simulations[n_in:]
        for bunch, idx in zip(OOT, np.array_split(np.arange(len(rest)), len(OOT))):
            sims = [rest[i] for i in idx]
            capture(self._time_oot, sims, None if phase_cut is None else phase_keeps(sims, phase_cut), True, bunch=bunch)
        simulations = simulations[:n_in]
    return _add(self, simulations, phase_cut=phase_cut, eta_cut=eta_cut, minbias=minbias)


def gather_rechit_energy(self, simulations, from_particle_hit_data=False):
    if not from_particle_hit_data:
        capture(self._time_store, simulations, None, self._time_minbias)
    return _gather(self, simulations, from_particle_hit_data)


def filter_simulations(self, simulations, phase_cut=None, eta_cut=None):
    if phase_cut is not None:
        phase_cut = draw_window(phase_cut)
        capture(self._time_store, simulations, phase_keeps(simulations, phase_cut), self._time_minbias)
    return _filter(self, simulations, phase_cut=phase_cut, eta_cut=eta_cut)


def gather_event_data(self, reduced_simulation):
    if self.include_tracks:
        raise NotImplementedError('hit times with include_tracks: track rows have no sensor')
    n_sensors = len(self.all_sensors_x)
    leak = hit_time.leak_energy(self._time_oot, n_sensors) if self._time_oot else None
    if leak is not None:                                 # raw GeV, before the builder's noise cut and calibration
        self.rechit_energy = self.rechit_energy + leak
    result = _event_data(self, reduced_simulation)
    time_all, charge_all = hit_time.hit_times(self._time_store + self._time_oot, n_sensors, PARAMS['threshold_fc'],
                                              PARAMS['noise_gev'], np.random, jitter=PARAMS['jitter'])
    idx = np.asarray(result['rechit_sensor_idx'], dtype=np.int64)
    result['rechit_time'] = time_all[idx]
    result['rechit_time_charge'] = charge_all[idx]
    if leak is not None:                                 # calibrated, like rechit_energy
        result['rechit_oot_energy'] = (leak * self.all_sensors_scale)[idx].astype(np.float32)
    return result


EG.reset, EG.add, EG._gather_rechit_energy, EG._filter_simulations, EG._gather_event_data = \
    reset, add, gather_rechit_energy, filter_simulations, gather_event_data

sys.argv = [BUILD_EVENTS] + sys.argv[1:3]
runpy.run_path(BUILD_EVENTS, run_name='__main__')
