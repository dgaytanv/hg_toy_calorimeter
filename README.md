## Warning
Not ready yet. Use with caution.

# A High Granularity Toy Calorimeter
The source code for a toy calorimeter to study particle reconstruction performance.

This fork of https://github.com/shahrukhqasim/hg_toy_calorimeter adds:
- hit times for every rec-hit, built the way CMSSW builds the HGCAL rec-hit time: section "Hit time" below;
- `ssl_interface/`: conversion of the toy events to the npz schema of `ssl_hep_np`, with its loader:
  `ssl_interface/README.md`.

This calorimeter was also employed in the following paper:
https://link.springer.com/article/10.1140/epjc/s10052-022-10665-7


Cite as:
```
@article{Qasim_2022,
	doi = {10.1140/epjc/s10052-022-10665-7},
	url = {https://doi.org/10.1140%2Fepjc%2Fs10052-022-10665-7},
	year = 2022,
	month = {aug},
	publisher = {Springer Science and Business Media {LLC}},
	volume = {82},
	number = {8},
	author = {Shah Rukh Qasim and Nadezda Chernyavskaya and Jan Kieseler and Kenneth Long and Oleksandr Viazlo and Maurizio Pierini and Raheel Nawaz},
	title = {End-to-end multi-particle reconstruction in high occupancy imaging calorimeters with graph neural networks},
	journal = {The European Physical Journal C}
}
```

## Install / Compile
The following docker container should be used to run the code in this repository:
https://hub.docker.com/r/shahrukhqasim2/g4calo. The repository has to be manually
compiled within the container. A container with a prebuilt package is coming soon.

The singularity `.def` and `Dockerfile` are both available in `containers/` directory if the container needs
to be built.

```
singularity pull docker://shahrukhqasim2/g4calo:latest
```
This step can take many hours to complete. As the result, a `.sif` file will be created in the
current directory. Login to the `.sif` container:
```
singularity shell g4calo_latest.sif
```
If at CERN, `EOS` and `AFS` can also be mounted as follows:
```
singularity shell -B /afs -B /eos g4calo_latest.sif
```
Check out the repository:
```
git clone https://github.com/shahrukhqasim/hg_toy_calorimeter.git
cd hg_toy_calorimeter
git submodule update --init --recursive
```

And build:
```
mkdir build
cd build
cmake ..
make -j
cd ..
```

## Generating data
As described in the paper, there are two steps to generate a datset with this calorimeter.
First individual simulations are generated which can be either single particle simulations
or proton-proton interactions. A dataset of simulations is generated and then to generate
events, simulations are sampled for every event without replacement.

### Setting environment:
To set the environment once the code has been built, the build directory should be added
to `PYTHONPATH`.
```
cd python/bin
export PYTHONPATH=$PYTHONPATH:`readlink -f ../../build/`:`readlink -f ../../build/lib/`:`readlink -f ../libs`
cd ../..
```

### Generate simulations
`run_simulation.py` script can be used to generate a set of simulations:
```
cd python/bin
python3 run_simulations.py minbias --cores=1
```
The simulations will be generated in `ra_pickles` format, as described here:
https://github.com/shahrukhqasim/ra_pickles

They are inherently stored as pickled numpy arrays and therefore, can be accessed
outside the container, by only installing the `ra_pickles` package via
`pip3 install ra-pickles`. The container should only be used for gerating events
and simulations.

### Generate events
`generate_events.py` script can be used to generate a set of events:
```
cd python/bin
python3 create_dataset.py example_datasets_config.ini test_dataset_cern
```

Here, first the HGCAL environment must have been sourced, to be able to
export data in `.djctd` format.

## Hit time (this fork)

This fork gives every rec-hit a time, built the way CMSSW builds the HGCAL rec-hit time. Upstream code changes only
in the generator (four files); the event builder runs upstream's `create_dataset.py` through runtime patches.

### Generator (`src/`, `include/`)
Every energy deposit in a sensor keeps the Geant4 global time of its step. Each simulation gets five more keys, with the
deposits summed per (particle, sensor, 1 ps time slice), as CMSSW's HGCAL sensitive detector does:
- `step_particle_id`: index of the simulation's particle (-1: no particle);
- `step_sensor_idx`;
- `step_deposit`: GeV, float32;
- `step_time`: ns since the collision, float32;
- `step_count_raw`: the number of Geant4 steps before the summing.

A particle started away from the origin (the `singlepart_face` gun) starts at its time of flight from the origin, so
the times of every simulation count from the collision. Rebuild after checking out this fork (see "Install /
Compile"); `run_simulations.py` is unchanged.

### Events (`hit_time/`)
`build_events_time.py <ini> <section>` runs `python/bin/create_dataset.py` unchanged, through runtime patches. It
writes the events in `ra_pickles` format instead of `.djctd` (through `build_events.py`), so it needs neither
DeepJetCore nor HGCalML. Each event is the EventGenerator's result dict, with the particle table attached as
`result['particles']`. Run it from `python/bin`, with the environment of "Setting environment". Every event also gets:
- `rechit_time`: ns; -99 where the hit has no valid time, as in the CMS files;
- `rechit_time_charge`: the cell's in-time charge, fC;
- with out-of-time pileup, `rechit_oot_energy`: the part of the hit energy from other bunch crossings.

The recipe is in `hit_time.py`. It follows CMSSW's HGCDigitizer, HGCFEElectronics and HGCalUncalibRecHitRecWeightsAlgo,
with the constants of the 300 um silicon sensors:
1. charge = deposit converted from keV to fC;
2. every collision has its own time and beam-line position: Gaussian, sigma 0.102 ns and 2.56 cm;
3. only deposits inside the in-time 25 ns bunch window count (after a 9 ns time-of-flight shift);
4. the time of arrival is when the time-ordered cumulative charge of the cell crosses the threshold, interpolated
   between the two deposits around it;
5. smearing: a Gaussian jitter of 5 ns x noise / charge, plus one common Gaussian offset per event (sigma 0.02 ns);
6. stored in steps of 0.0244 ns.

The default threshold is 40 fC instead of CMSSW's 12 fC: the toy's cells are larger, and 40 fC gives HGCAL's fraction
of hits with a time.

Out-of-time pileup: with `time_oot_bunches=cms` (bunch crossings -3 to +3) or a list such as `-2,-1,1`, the
minimum-bias simulations beyond the in-time pileup are spread over those bunch crossings. Set
`num_pu` = in-time pileup x (1 + number of out-of-time crossings). These collisions have no truth. Their deposits move
by 25 ns per crossing. They add energy to the in-time hits with the HGCROC pulse-shape weights, and enter the time of
arrival when they fall in the in-time window.

Settings are ini keys of the section: `time_threshold_fc`, `time_collision_sigma_ns`, `time_collision_sigma_z_cm`,
`time_noise_gev` (the cell noise for the jitter; default `noise_sigma`), `time_jitter`, `time_oot_bunches`. See
`hit_time/example_time_build.ini`.

### Checks (`hit_time/`)
- `test_hit_time.py`: known-answer tests of the recipe (numpy only).
- `verify_time_sims.py <simulation folder> <simtype>`: the step deposits sum to `hit_particle_deposit` and
  `rechit_energy`, and no time comes before the calorimeter front.
- `verify_time_events.py <event folder> <calibration .bin> <threshold fC>`: per event, the time columns against the
  energies and the threshold.
- `threshold_scan.py <event folder>`: the fraction of particle hits with a time at other thresholds, from the stored
  charges, and the threshold that gives HGCAL's 25.2 %.

### Measured
PU-free events with 60 particles each, in the paper's geometry, at 40 fC:
- 22.7 % of all hits have a time, holding 83 % of the energy (two sets of 5,000 events);
- 25.4 % of the particle hits have a time (1,000 events). HGCAL: 25.2 % of the particle hits of 500 CMS tau events.
  The toy would match at 40.4 fC.

The fraction depends on the cell sizes. With another geometry, run `threshold_scan.py` and set `time_threshold_fc`.
The pileup options were tested on small samples only.

* Events export in ra_pickles format is coming soon.