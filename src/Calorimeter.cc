//
// Created by Shah Rukh Qasim on 07.01.22.
//

#include "Calorimeter.hh"

#include "B4DetectorConstruction.hh"
#include "B4aActionInitialization.hh"
#include <string>
#include <cmath>
#include <functional>
#include <unordered_map>

#include "defines.h"


#ifdef G4MULTITHREADED
#undef G4MULTITHREADED
#endif

#ifdef G4MULTITHREADED
#include "G4MTRunManager.hh"
#else

#include "G4RunManager.hh"

#endif

#include<string>

#include "G4UImanager.hh"
#include "G4UIcommand.hh"
#include "FTFP_BERT.hh"
#include "QGSP_FTFP_BERT.hh"

#include "Randomize.hh"

#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "G4RandomTools.hh"
#include "B4DetectorConstructionFP.hh"

//limiter stuff

#include "G4PhysicsListHelper.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4UserSpecialCuts.hh"
#include "G4StepLimiter.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"

#include "B4PartGeneratorBase.hh"
#include "B4aEventAction.hh"
#include "PrimariesGenerator.hh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <G4RNGHelper.hh>

namespace py = pybind11;
using namespace py::literals;



G4RunManager* runManager;
B4aEventAction* eventAction;
G4UImanager *ui_manager;
PrimariesGenerator *primariesGenerator;
B4DetectorConstruction * detector;
bool collect_full_data;
CLHEP::MTwistEngine *randomEngine;

void wrap_up() {
    delete runManager;
}
void initialize_test(long rseed, std::string detector_specs) {
    rseed++;

    B4PartGeneratorBase::seedsoffset_ = 800 * rseed;
    G4cout << "random seed " << rseed << G4endl;

    runManager = new G4RunManager;

    Json::Value root(detector_specs);

    auto layers_upper = root["layers"];

}

void fix_seeding() {
    long all_seeds[10];

    for(int i=0;i<10;i++) {
        double seed = randomEngine->flat();
        long seed_l = seed*1000000000;
        all_seeds[i] = seed_l;
    }

    CLHEP::HepRandom::setTheSeeds(all_seeds);
    G4Random::setTheSeeds(all_seeds);
//    std::cout<<"X2 Random seed is "<<seed_l<<std::endl;

//    auto helper = G4RNGHelper::GetInstance();
//    helper->Clear();//To remove any seed from previous run
//    //Fill seeds one by one, use:
//    helper->AddOneSeed( seed_l );
////    Or fill many:
////    assert( size(seedsArray) == numEvents*number_seeds_per_event );
////    helper->Fill( seedsArray , numEvents , numEvents , number_seeds_per_event);
//    //Next line is very important, to avoid retriggers of fills from kernel with default strategy/
//    //Communicate how many events are ready
////    G4MTRunManager::nSeedsFilled = 1;
}

void initialize(std::string detector_specs, std::string pythiadata, bool collect_full_data_, int rseed_0,
                int rseed_1, int rseed_2, int rseed_3) {
//    rseed++;

//    B4PartGeneratorBase::seedsoffset_ = 800 * rseed;
//    G4cout << "random seed " << rseed << G4endl;

    randomEngine = new CLHEP::MTwistEngine(rseed_0);
//    fix_seeding();
//    long seeds10[10];


    long seeds[4] = {rseed_0, rseed_1, rseed_2, rseed_3};

    CLHEP::HepRandom::setTheSeeds(seeds);
    G4Random::setTheSeeds(seeds);

    runManager = new G4RunManager;

    Json::Value root(detector_specs);
    Json::Reader reader;
    reader.parse(detector_specs, root);

    detector = new B4DetectorConstruction(root);
    runManager->SetUserInitialization(detector);

    auto physicsList = new FTFP_BERT;
    physicsList->RegisterPhysics(new G4StepLimiterPhysics());
    runManager->SetUserInitialization(physicsList);

    collect_full_data=collect_full_data_;
    eventAction = new B4aEventAction(std::string(), collect_full_data_);
    primariesGenerator = new PrimariesGenerator(pythiadata, rseed_0, rseed_1, rseed_2, rseed_3);
    auto actionInitialization = new B4aActionInitialization(detector, eventAction, primariesGenerator);
    runManager->SetUserInitialization(actionInitialization);


    // Get the pointer to the User Interface manager
    ui_manager = G4UImanager::GetUIpointer();

    ui_manager->ApplyCommand(std::string("/run/initialize"));
    ui_manager->ApplyCommand(std::string("/run/printProgress 100"));
}

py::dict collect() {
    const std::vector<double>& rechit_energy = eventAction->getRechitEnergy();
    const std::vector<double>& rechit_x= eventAction->getRechitX();
    const std::vector<double>& rechit_y= eventAction->getRechitY();
    const std::vector<double>& rechit_z= eventAction->getRechitZ();
    const std::vector<int>& rechit_stupid_pid= eventAction->getRechitStupidPid();

    std::vector<int> rechit_idx;
    std::vector<double> rechit_energy_2;
    std::vector<double> rechit_x_2;
    std::vector<double> rechit_y_2;
    std::vector<double> rechit_z_2;
    std::vector<int> rechit_stupid_pid_2;

    for (int i=0;i<rechit_energy.size();i++) {
        if (rechit_energy[i]>0) {
            rechit_idx.push_back(i);
            rechit_energy_2.push_back(rechit_energy[i]);
            rechit_x_2.push_back(rechit_x[i]);
            rechit_y_2.push_back(rechit_y[i]);
            rechit_z_2.push_back(rechit_z[i]);
            rechit_stupid_pid_2.push_back(rechit_stupid_pid[i]);
        }
    }

    py::array rechitEnergy = py::cast(rechit_energy_2);
    py::array rechitX = py::cast(rechit_x_2);
    py::array rechitY = py::cast(rechit_y_2);
    py::array rechitZ = py::cast(rechit_z_2);
    py::array rechitIdx = py::cast(rechit_idx);
    py::array rechitStupidPid = py::cast(rechit_stupid_pid_2);

    auto&hits_particles = eventAction->getHitsParticles();
    auto&hits_deposits = eventAction->getHitsDeposits();
    auto&hits_sensor_idx = eventAction->getHitsParticlesSensorIdx();

    std::vector<int> hits_particles_2;
    std::vector<double> hits_deposits_2;
    std::vector<int> hits_sensor_idx_2;

    std::unordered_map<std::string, int> ps_to_idx;
    int max_k = 0;
    for(int i=0;i < hits_particles.size();i++) {
        if(hits_deposits[i]==0) {
            continue;
        }
        std::string k = std::to_string(hits_particles[i]) + std::string("_") + std::to_string(hits_sensor_idx[i]);
        if (ps_to_idx.find(k) != ps_to_idx.end()) {
            hits_deposits_2[ps_to_idx[k]] += hits_deposits[i];
        }
        else {
            hits_particles_2.push_back(hits_particles[i]);
            hits_deposits_2.push_back(hits_deposits[i]);
            hits_sensor_idx_2.push_back(hits_sensor_idx[i]);
            ps_to_idx[k] = max_k;
            max_k += 1;
        }
    }

    py::array np_hits_particles_2  = py::cast(hits_particles_2);
    py::array np_hits_deposits_2  = py::cast(hits_deposits_2);
    py::array np_hits_sensor_idx_2  = py::cast(hits_sensor_idx_2);

    py::array hitsParticles = py::cast(eventAction->getHitsParticles());
    py::array hitsDeposits = py::cast(eventAction->getHitsDeposits());
    py::array hitsSensorIndex = py::cast(eventAction->getHitsParticlesSensorIdx());

    std::vector<int> a;
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);

    py::array particles_vertex_position_x = py::cast(eventAction->particles_caught.particles_vertex_position_x);
    py::array particles_vertex_position_y = py::cast(eventAction->particles_caught.particles_vertex_position_y);
    py::array particles_vertex_position_z = py::cast(eventAction->particles_caught.particles_vertex_position_z);

    py::array particles_momentum_direction_x = py::cast(eventAction->particles_caught.particles_momentum_direction_x);
    py::array particles_momentum_direction_y = py::cast(eventAction->particles_caught.particles_momentum_direction_y);
    py::array particles_momentum_direction_z = py::cast(eventAction->particles_caught.particles_momentum_direction_z);

    py::array particles_first_active_impact_position_x = py::cast(eventAction->particles_caught.particles_first_active_impact_position_x);
    py::array particles_first_active_impact_position_y = py::cast(eventAction->particles_caught.particles_first_active_impact_position_y);
    py::array particles_first_active_impact_position_z = py::cast(eventAction->particles_caught.particles_first_active_impact_position_z);
    py::array particles_first_active_impact_sensor_idx = py::cast(eventAction->particles_caught.particles_first_active_impact_sensor_idx);
    py::array particles_first_impact_kinetic_energy = py::cast(eventAction->particles_caught.particles_first_impact_kinetic_energy);

    py::array particles_first_active_impact_momentum_direction_x = py::cast(eventAction->particles_caught.particles_first_active_impact_momentum_direction_x);
    py::array particles_first_active_impact_momentum_direction_y = py::cast(eventAction->particles_caught.particles_first_active_impact_momentum_direction_y);
    py::array particles_first_active_impact_momentum_direction_z = py::cast(eventAction->particles_caught.particles_first_active_impact_momentum_direction_z);

    for(auto i =0;i<eventAction->particles_caught.particles_kinetic_energy.size();i++) {
        std::cout<<"V3 K1 K2 DepAll "<<eventAction->particles_caught.particles_kinetic_energy[i]<<" "<<eventAction->particles_caught.particles_first_impact_kinetic_energy[i]<<" "<<eventAction->particles_caught.particles_total_energy_deposited_all[i]<<std::endl;
    }

    py::array particles_kinetic_energy = py::cast(eventAction->particles_caught.particles_kinetic_energy);
    py::array particles_pdgid = py::cast(eventAction->particles_caught.particles_pdgid);
    py::array particles_total_energy_deposited_active = py::cast(eventAction->particles_caught.particles_total_energy_deposited_active);
    py::array particles_total_energy_deposited_all = py::cast(eventAction->particles_caught.particles_total_energy_deposited_all);
    py::array particles_parent_id = py::cast(eventAction->particles_caught.particles_parent_idx);

    std::string simtype = primariesGenerator->getSimTypeAsString();


//    delete ev;
    py::dict d = py::dict(
            "simtype"_a =simtype,
            "rechit_energy"_a =rechitEnergy,
                          "rechit_x"_a = rechitX,
                          "rechit_y"_a = rechitY,
                          "rechit_z"_a = rechitZ,
                          "rechit_idx"_a = rechitIdx,
                          "rechit_stupid_pid"_a = rechitStupidPid,

//                          "rechit_particle_id"_a = hitsParticles,
//                          "rechit_particle_deposit"_a = hitsDeposits,
//                          "rechit_particle_sensor_idx"_a = hitsSensorIndex,

                          "hit_particle_id"_a = np_hits_particles_2,
                          "hit_particle_deposit"_a = np_hits_deposits_2,
                          "hit_particle_sensor_idx"_a = np_hits_sensor_idx_2,

//                          "particle_positions_x"_a = (py::array) py::cast(eventAction->positions_tracked_x),
//                          "particle_positions_y"_a = (py::array) py::cast(eventAction->positions_tracked_y),
//                          "particle_positions_z"_a = (py::array) py::cast(eventAction->positions_tracked_z),
//                          "particle_positions_particle_id"_a = (py::array) py::cast(eventAction->positions_tracked_particle_index),

                          "particles_vertex_position_x"_a = particles_vertex_position_x,
                          "particles_vertex_position_y"_a = particles_vertex_position_y,
                          "particles_vertex_position_z"_a = particles_vertex_position_z,
                          "particles_momentum_direction_x"_a = particles_momentum_direction_x,
                          "particles_momentum_direction_y"_a = particles_momentum_direction_y,
                          "particles_momentum_direction_z"_a = particles_momentum_direction_z,

                          "particles_first_active_impact_position_x"_a = particles_first_active_impact_position_x,
                          "particles_first_active_impact_position_y"_a = particles_first_active_impact_position_y,
                          "particles_first_active_impact_position_z"_a = particles_first_active_impact_position_z,
                          "particles_first_active_impact_sensor_idx"_a = particles_first_active_impact_sensor_idx,
                          "particles_first_impact_active_energy"_a = particles_first_impact_kinetic_energy,

                           "particles_first_active_impact_momentum_direction_x"_a = particles_first_active_impact_momentum_direction_x,
                           "particles_first_active_impact_momentum_direction_y"_a = particles_first_active_impact_momentum_direction_y,
                           "particles_first_active_impact_momentum_direction_z"_a = particles_first_active_impact_momentum_direction_z,


    "particles_kinetic_energy"_a = particles_kinetic_energy,
                          "particles_pdgid"_a = particles_pdgid,
                          "particles_total_energy_deposited_active"_a = particles_total_energy_deposited_active,
                          "particles_total_energy_deposited_all"_a = particles_total_energy_deposited_all,
                          "particles_parent_idx"_a = particles_parent_id
    );

    {   // HIT TIME: the deposits summed per (particle, sensor, 1 ps time slice), the time slice of CMSSW's HGCSD
        // (TimeSliceUnit 0.001 ns). Keys: step_particle_id (the index of hit_particle_id; -1 = a deposit attributed
        // to no particle), step_sensor_idx, step_deposit (GeV, float32), step_time (ns, float32: exact to < 4 ps
        // below 34 ns). Everything later than 1 microsecond goes into the 1 microsecond slice.
        struct Key {
            int particle, sensor; long long slice;
            bool operator==(const Key &o) const { return particle == o.particle && sensor == o.sensor && slice == o.slice; }
        };
        struct KeyHash {
            size_t operator()(const Key &k) const {
                size_t h = std::hash<long long>()(((long long) k.particle << 32) ^ (long long) (unsigned int) k.sensor);
                return h ^ (std::hash<long long>()(k.slice) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
            }
        };
        std::unordered_map<Key, size_t, KeyHash> slot;
        std::vector<int> s_part, s_idx;
        std::vector<double> s_dep_sum;
        std::vector<float> s_time;
        const auto &sp = eventAction->steps_particle_idx;
        const auto &ss = eventAction->steps_sensor_idx;
        const auto &sd = eventAction->steps_deposit;
        const auto &st = eventAction->steps_time;
        for (size_t i = 0; i < ss.size(); i++) {
            long long slice = (long long) std::floor(st[i] * 1000.0);   // ps
            if (slice < 0) slice = 0;
            if (slice > 1000000LL) slice = 1000000LL;
            Key key{sp[i], ss[i], slice};
            auto found = slot.find(key);
            if (found == slot.end()) {
                slot[key] = s_idx.size();
                s_part.push_back(sp[i]);
                s_idx.push_back(ss[i]);
                s_dep_sum.push_back(sd[i]);
                s_time.push_back((float) (slice * 0.001));
            } else {
                s_dep_sum[found->second] += sd[i];
            }
        }
        std::vector<float> s_dep(s_dep_sum.begin(), s_dep_sum.end());
        d["step_particle_id"] = (py::array) py::cast(s_part);
        d["step_sensor_idx"] = (py::array) py::cast(s_idx);
        d["step_deposit"] = (py::array) py::cast(s_dep);
        d["step_time"] = (py::array) py::cast(s_time);
        d["step_count_raw"] = (long long) ss.size();
    }

    if (collect_full_data) {
        for (auto& it: eventAction->result_arrays_int) {
            py::array np = py::cast(*it.second);
            d[it.first.c_str()] = np;
        }
        for (auto& it: eventAction->result_arrays_double) {
            py::array np = py::cast(*it.second);
            d[it.first.c_str()] = np;
        }
    }


    return d;
}

py::dict simulate_pu() {
    if (ui_manager == nullptr) {
        G4cout<<"Forgot to call initialize?\n";
        throw std::runtime_error("Forgot to call initialize?");
    }

    primariesGenerator->SetNextToPU();
    ui_manager->ApplyCommand(std::string("/run/beamOn ") + std::to_string(1));

    return collect();
}


pybind11::dict generate_pu_without_sim() {
    if (ui_manager == nullptr) {
        G4cout<<"Forgot to call initialize?\n";
        throw std::runtime_error("Forgot to call initialize?");
    }

    primariesGenerator->SetNextToPU();
    G4PrimaryVertex *vertex = primariesGenerator->GeneratePUParticles();

    int num = vertex->GetNumberOfParticle();

    std::vector<double> kinetic_energy;
    std::vector<double> momentum_direction_x;
    std::vector<double> momentum_direction_y;
    std::vector<double> momentum_direction_z;

    for (int i = 0;i<num;i++) {
        auto particle = vertex->GetPrimary(i);
        kinetic_energy.push_back(particle->GetKineticEnergy());
        momentum_direction_x.push_back(particle->GetMomentumDirection().x());
        momentum_direction_y.push_back(particle->GetMomentumDirection().y());
        momentum_direction_z.push_back(particle->GetMomentumDirection().z());
    }

    py::dict d = py::dict(
            "particle_kinetic_energy"_a =kinetic_energy,
            "momentum_direction_x"_a =momentum_direction_x,
            "momentum_direction_y"_a =momentum_direction_y,
            "momentum_direction_z"_a =momentum_direction_z
    );

    delete vertex;
    return d;
}

py::dict simulate_qqbar2ttbar() {
    if (ui_manager == nullptr) {
        G4cout<<"Forgot to call initialize?\n";
        throw std::runtime_error("Forgot to call initialize?");
    }

    primariesGenerator->SetNextTo_qqbar_to_ttbar();
    ui_manager->ApplyCommand(std::string("/run/beamOn ") + std::to_string(1));

    return collect();
}



py::dict simulate_particle(double position_x, double position_y, double position_z, double direction_x,
                           double direction_y,
                           double direction_z, int pdgid, double energy) {
    if (ui_manager == nullptr) {
        G4cout<<"Forgot to call initialize?\n";
        throw std::runtime_error("Forgot to call initialize?");
    }



    primariesGenerator->SetNextToParticle(position_x,position_y,position_z,direction_x,direction_y,direction_z,pdgid,energy);
    ui_manager->ApplyCommand(std::string("/run/beamOn ") + std::to_string(1));
    return collect();
}


py::dict get_sensor_data() {
    if (ui_manager == nullptr) {
        G4cout<<"Forgot to call initialize?\n";
        throw std::runtime_error("Forgot to call initialize?");
    }

    const auto &activesensors = detector->getActiveSensors();

    std::vector<double> sensors_x;
    std::vector<double> sensors_y;
    std::vector<double> sensors_z;
    std::vector<double> sensors_area;
    std::vector<int> sensors_active_layer_num;
    std::vector<double> sensors_pre_absorber_thickness;
    std::vector<double> sensors_thickness;

    std::vector<double> sensors_deta;
    std::vector<double> sensors_dphi;

    for(const auto & activesensor : *activesensors) {
        sensors_x.push_back(activesensor->getPosx());
        sensors_y.push_back(activesensor->getPosy());
        sensors_z.push_back(activesensor->getPosz());
        sensors_area.push_back(activesensor->getArea());
        sensors_active_layer_num.push_back(activesensor->getLayer());
        sensors_pre_absorber_thickness.push_back(activesensor->getPreAbsorberThickness());
        sensors_thickness.push_back(activesensor->getThickness());
        sensors_deta.push_back(activesensor->getDeta());
        sensors_dphi.push_back(activesensor->getDphi());
    }

    py::dict d = py::dict("sensors_x"_a = (py::array)py::cast(sensors_x),
                          "sensors_y"_a = (py::array) py::cast(sensors_y),
                          "sensors_z"_a = (py::array) py::cast(sensors_z),
                          "sensors_area"_a = (py::array) py::cast(sensors_area),
                          "sensors_active_layer_num"_a = (py::array) py::cast(sensors_active_layer_num),
                          "sensors_pre_absorber_thickness"_a = (py::array) py::cast(sensors_pre_absorber_thickness),
                          "sensors_thickness"_a = (py::array) py::cast(sensors_thickness),
                          "sensors_deta"_a = (py::array) py::cast(sensors_deta),
                          "sensors_dphi"_a = (py::array) py::cast(sensors_dphi));

    return d;

}

pybind11::dict dict_check(int number_of_events, long rseed, std::string detector_specs) {
//    py::dict d;
    std::vector<float> rechitEnergy;
    rechitEnergy.push_back(1.0);
    rechitEnergy.push_back(2.0);
    rechitEnergy.push_back(3.0);


    std::vector<float> rechitDummy;
    rechitDummy.push_back(1.0);
    rechitDummy.push_back(2.2);
    rechitDummy.push_back(3.0);

    std::string name("dummy_dummy");

    py::dict d = py::dict("rechit_energy"_a = rechitEnergy,
                          py::arg(name.c_str()) = rechitDummy
                          );

    d["damn"] = rechitDummy;

    return d;
}
