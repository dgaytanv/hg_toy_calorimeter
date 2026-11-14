//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// $Id: B4aEventAction.cc 100946 2016-11-03 11:28:08Z gcosmo $
//
/// \file B4aEventAction.cc
/// \brief Implementation of the B4aEventAction class

#include "B4aEventAction.hh"
#include "B4RunAction.hh"
#include "B4Analysis.hh"

#include "G4RunManager.hh"
#include "G4Event.hh"
#include "G4UnitsTable.hh"

#include "Randomize.hh"
#include <iomanip>
#include "G4INCLRandom.hh"
#include "B4PrimaryGeneratorAction.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aEventAction::B4aEventAction(std::string output_folder_name, bool collect_full_data)
        : G4UserEventAction(),
          fEnergyAbs(0.),
          fEnergyGap(0.),
          fTrackLAbs(0.),
          fTrackLGap(0.),
          generator_(0),
          nsteps_(0) {
    this->output_bin_folder = output_folder_name;

    this->collect_full_data = collect_full_data;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aEventAction::~B4aEventAction() {}

void SOAParticles::add(double vertex_position_x, double vertex_position_y, double vertex_position_z,
                       double momentum_direction_x, double momentum_direction_y, double momentum_direction_z,
                       double kinetic_energy, int pdgid, int trackid, bool tagged, int parent_idx) {

    trackid_to_idx[trackid] = (int) particles_vertex_position_x.size();
    particles_vertex_position_x.push_back(vertex_position_x);
    particles_vertex_position_y.push_back(vertex_position_y);
    particles_vertex_position_z.push_back(vertex_position_z);
    particles_momentum_direction_x.push_back(momentum_direction_x);
    particles_momentum_direction_y.push_back(momentum_direction_y);
    particles_momentum_direction_z.push_back(momentum_direction_z);
    particles_kinetic_energy.push_back(kinetic_energy);
    particles_pdgid.push_back(pdgid);
    particles_total_energy_deposited_active.push_back(0.);
    particles_total_energy_deposited_all.push_back(0.);
    particles_tagged.push_back(tagged);
    particles_parent_idx.push_back(parent_idx);

    particles_first_impact_kinetic_energy.push_back(-1);
    particles_first_active_impact_position_x.push_back(-1);
    particles_first_active_impact_position_y.push_back(-1);
    particles_first_active_impact_position_z.push_back(-1);
    particles_first_active_impact_sensor_idx.push_back(-1);

    particles_first_active_impact_momentum_direction_x.push_back(-1);
    particles_first_active_impact_momentum_direction_y.push_back(-1);
    particles_first_active_impact_momentum_direction_z.push_back(-1);

}

void SOAParticles::clear() {
    particles_vertex_position_x.clear();
    particles_vertex_position_y.clear();
    particles_vertex_position_z.clear();

    particles_momentum_direction_x.clear();
    particles_momentum_direction_y.clear();
    particles_momentum_direction_z.clear();

    particles_kinetic_energy.clear();
    particles_pdgid.clear();

    particles_parent_idx.clear();
    particles_total_energy_deposited_active.clear();
    particles_total_energy_deposited_all.clear();

    trackid_to_idx.clear();

    particles_first_impact_kinetic_energy.clear();
    particles_first_active_impact_position_x.clear();
    particles_first_active_impact_position_y.clear();
    particles_first_active_impact_position_z.clear();
    particles_first_active_impact_sensor_idx.clear();

    particles_first_active_impact_momentum_direction_x.clear();
    particles_first_active_impact_momentum_direction_y.clear();
    particles_first_active_impact_momentum_direction_z.clear();

}

void B4aEventAction::accumulateStepData(G4VPhysicalVolume *volume, const G4Step *step) {
    G4Track* track = step->GetTrack();
    auto pos1 = track->GetVertexPosition();
    auto pos2 = track->GetPosition();

    auto calo_start_z = detector_->getCaloStartZ();

    double limit = calo_start_z;
    limit = limit*1.0;

    int particle_index = -1;

    int parent_idx = -1;
    if(particles_caught.trackid_to_idx.find(track->GetParentID()) != particles_caught.trackid_to_idx.end()) {
        parent_idx = particles_caught.trackid_to_idx[track->GetParentID()];
    }

    bool catch_position=false;
    if (particles_caught.trackid_to_idx.find(track->GetTrackID()) == particles_caught.trackid_to_idx.end()) {
        if (pos1.getZ() <= limit) {
            particle_index = (int) particles_caught.particles_vertex_position_x.size();
            tracks_buckets[track->GetTrackID()] = particle_index;

            std::cout<<"-1- Inserting " << particle_index<<" with parent "<<parent_idx<<" and energy "<<track->GetKineticEnergy()/1000<<" GeV and pdgid "<< track->GetParticleDefinition()->GetPDGEncoding()<<std::endl;
//            std::cout<<"Position "<<pos1.x()<<" "<<pos1.y()<<" "<<pos1.z()<<std::endl;
//            std::cout<<"Direction "<<track->GetMomentumDirection().x()<<" "<<track->GetMomentumDirection().y()<<" "<<track->GetMomentumDirection().z()<<std::endl;
//            std::cout<<"PDGID "<<track->GetParticleDefinition()->GetPDGEncoding()<<std::endl;

            particles_caught.add(
                    pos1.x(),
                    pos1.y(),
                    pos1.z(),
                    track->GetMomentumDirection().x(),
                    track->GetMomentumDirection().y(),
                    track->GetMomentumDirection().z(),
                    track->GetKineticEnergy() / 1000.f,
                    track->GetParticleDefinition()->GetPDGEncoding(),
                    track->GetTrackID(),
                    true,
                    parent_idx
                    );

//            positions_tracked_particle_index.push_back(particle_index);
//            positions_tracked_x.push_back(pos1.x());
//            positions_tracked_y.push_back(pos1.y());
//            positions_tracked_z.push_back(pos1.z());

            catch_position = true;
        }
    } else {
        particle_index = particles_caught.trackid_to_idx[track->GetTrackID()];
        catch_position = true;

        if (step->GetTotalEnergyDeposit() != 0) {
            if (particles_caught.particles_first_impact_kinetic_energy[particle_index] == -1) {
                particles_caught.particles_first_impact_kinetic_energy[particle_index] =
                        track->GetKineticEnergy() / 1000.;
            }
        }

    }

    if(particle_index==-1) {
        if (tracks_buckets.find(track->GetParentID()) != tracks_buckets.end()) {
            particle_index = (int) tracks_buckets[track->GetParentID()];
            tracks_buckets[track->GetTrackID()] = particle_index;
        }
    }

//    if(particle_index!=-1) {
//        positions_tracked_particle_index.push_back(particle_index);
//        positions_tracked_x.push_back(pos2.x());
//        positions_tracked_y.push_back(pos2.y());
//        positions_tracked_z.push_back(pos2.z());
//    }

    const auto &activesensors = detector_->getActiveSensors();

    bool issensor = true;
    nsteps_++;

    G4StepPoint *preStepPoint = step->GetPreStepPoint();
    G4TouchableHistory *theTouchable =
            (G4TouchableHistory *) (preStepPoint->GetTouchable());
    auto volume2 = theTouchable->GetVolume();
    G4int copyNo =volume2->GetCopyNo();

    auto indexedSensorContainers = detector_->getIndexedSensorContainers();
    int idx = -1;

    if (indexedSensorContainers->find(volume2->GetInstanceID()) != indexedSensorContainers->end()) {
        idx = (indexedSensorContainers)->at(volume2->GetInstanceID()).at(copyNo)->getGlobalDetID();
    }

    auto energy = step->GetTotalEnergyDeposit() / 1000;

    total_deposit_all += energy;

    bool is_active=true;
    if (idx == -1) {
        is_active = false;//not active volume
        total_deposit_active += energy;
    }

    if (particle_index != -1 and energy != 0) {
        if(particles_caught.particles_first_active_impact_position_x[particle_index] == -1) {
                double sensor_pos_x = (*detector_->getActiveSensors())[idx]->getPosx();
                double sensor_pos_y = (*detector_->getActiveSensors())[idx]->getPosy();
                double sensor_pos_z = (*detector_->getActiveSensors())[idx]->getPosz();

                int tmp = -1;
                if (particles_caught.trackid_to_idx.find(track->GetTrackID()) != particles_caught.trackid_to_idx.end()) {
                    tmp = particles_caught.trackid_to_idx[track->GetTrackID()];
                }


                particles_caught.particles_first_active_impact_position_x[particle_index] = pos2.x();
                particles_caught.particles_first_active_impact_position_y[particle_index] = pos2.y();
                particles_caught.particles_first_active_impact_position_z[particle_index] = pos2.z();
                particles_caught.particles_first_active_impact_sensor_idx[particle_index] = idx;

                particles_caught.particles_first_active_impact_momentum_direction_x[particle_index] = track->GetMomentumDirection().x();
                particles_caught.particles_first_active_impact_momentum_direction_y[particle_index] = track->GetMomentumDirection().y();
                particles_caught.particles_first_active_impact_momentum_direction_z[particle_index] = track->GetMomentumDirection().z();
        }

        if (is_active) {
            if(particles_caught.particles_first_active_impact_sensor_idx[particle_index] == -1) {
                particles_caught.particles_first_active_impact_sensor_idx[particle_index] = idx;
            }

            particles_caught.particles_total_energy_deposited_active[particle_index] =
                    particles_caught.particles_total_energy_deposited_active[particle_index] + energy;


            hits_particles_id.push_back(particle_index);
            hits_particles_deposits.push_back(energy);
            hits_particles_sensor_idx.push_back(idx);
        }

        particles_caught.particles_total_energy_deposited_all[particle_index] =
                particles_caught.particles_total_energy_deposited_all[particle_index] + energy;

    }

    if (idx != -1) {
        rechit_energy_.at(idx) += energy;
        rechit_stupid_pid.at(idx) = particle_index;
        if (energy > 0) {   // HIT TIME: as CMSSW's CaloSD, the track's global time at the step
            steps_particle_idx.push_back(particle_index);
            steps_sensor_idx.push_back(idx);
            steps_deposit.push_back(energy);
            steps_time.push_back(track->GetGlobalTime() / CLHEP::ns);
        }
    }

    if (this->collect_full_data) {
        result_arrays_int["fulldata_trackid"]->push_back(track->GetTrackID());
        result_arrays_int["fulldata_parent_trackid"]->push_back(track->GetParentID());
        result_arrays_int["fulldata_particle_id"]->push_back(particle_index);
        result_arrays_double["fulldata_step_energy"]->push_back(step->GetTotalEnergyDeposit());
        result_arrays_double["fulldata_track_vertex_x"]->push_back(pos1.x());
        result_arrays_double["fulldata_track_vertex_y"]->push_back(pos1.y());
        result_arrays_double["fulldata_track_vertex_z"]->push_back(pos1.z());
        result_arrays_double["fulldata_track_pos_x"]->push_back(pos2.x());
        result_arrays_double["fulldata_track_pos_y"]->push_back(pos2.y());
        result_arrays_double["fulldata_track_pos_z"]->push_back(pos2.z());
        result_arrays_double["fulldata_momentum_direction_x"]->push_back(track->GetMomentumDirection().x());
        result_arrays_double["fulldata_momentum_direction_y"]->push_back(track->GetMomentumDirection().y());
        result_arrays_double["fulldata_momentum_direction_z"]->push_back(track->GetMomentumDirection().z());
        result_arrays_double["fulldata_sensor_id"]->push_back(idx);
        result_arrays_double["fulldata_track_kinetic_energy"]->push_back(track->GetKineticEnergy());
        result_arrays_double["fulldata_track_total_energy"]->push_back(track->GetTotalEnergy());
    }
}


void B4aEventAction::BeginOfEventAction(const G4Event * /*event*/) {
    // initialisation per event
    fEnergyAbs = 0.;
    fEnergyGap = 0.;
    fTrackLAbs = 0.;
    fTrackLGap = 0.;
    clear();
    totalen_ = 0;
    nsteps_ = 0;
    //set generator stuff
//random particle
    //random energy
    /*
    do this in the generator
    */

    //
    //

    result_arrays_double.clear();
    result_arrays_int.clear();

    const auto &activesensors = detector_->getActiveSensors();

    rechit_absorber_energy_ = std::vector<double>(activesensors->size(), 0);//reset
    rechit_energy_ = std::vector<double>(activesensors->size(), 0);//reset
    rechit_stupid_pid = std::vector<int>(activesensors->size(), 0);//reset
    rechit_x_.resize(activesensors->size(), 0);
    rechit_y_.resize(activesensors->size(), 0);
    rechit_z_.resize(activesensors->size(), 0);
    rechit_layer_.resize(activesensors->size(), 0);
    rechit_phi_.resize(activesensors->size(), 0);
    rechit_eta_.resize(activesensors->size(), 0);
    rechit_vxy_.resize(activesensors->size(), 0);
    rechit_detid_.resize(activesensors->size(), 0);
    rechit_idx__.resize(activesensors->size(), 0);

    for (size_t i = 0; i < activesensors->size(); i++) {
        rechit_x_.at(i) = activesensors->at(i)->getPosx();//eta
        rechit_y_.at(i) = activesensors->at(i)->getPosy();//phi
        rechit_z_.at(i) = activesensors->at(i)->getPosz();//z
        rechit_layer_.at(i) = activesensors->at(i)->getLayer();
        rechit_phi_.at(i) = activesensors->at(i)->getPhi();//0
        rechit_eta_.at(i) = activesensors->at(i)->getEta();//lengthz
        rechit_vxy_.at(i) = activesensors->at(i)->getArea();//0
        rechit_detid_.at(i) = activesensors->at(i)->getGlobalDetID();
        rechit_idx__.at(i) = i;
    }

    tracks_buckets.clear();
    particles_caught.clear();

    particle_index_not_found = 0;
    total_deposit_active = 0;
    total_deposit_all = 0;

    hits_particles_id.clear();
    hits_particles_deposits.clear();
    hits_particles_sensor_idx.clear();
    steps_particle_idx.clear();
    steps_sensor_idx.clear();
    steps_deposit.clear();
    steps_time.clear();


    if (this->collect_full_data) {
        result_arrays_int["fulldata_trackid"] = std::shared_ptr<std::vector<int>>(new std::vector<int>());
        result_arrays_int["fulldata_parent_trackid"] = std::shared_ptr<std::vector<int>>(new std::vector<int>());
        result_arrays_double["fulldata_step_energy"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
        result_arrays_double["fulldata_track_vertex_x"] = std::shared_ptr<std::vector<double>>(
                new std::vector<double>());
        result_arrays_double["fulldata_track_vertex_y"] = std::shared_ptr<std::vector<double>>(
                new std::vector<double>());
        result_arrays_double["fulldata_track_vertex_z"] = std::shared_ptr<std::vector<double>>(
                new std::vector<double>());
        result_arrays_double["fulldata_track_pos_x"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
        result_arrays_double["fulldata_track_pos_y"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
        result_arrays_double["fulldata_track_pos_z"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
        result_arrays_double["fulldata_momentum_direction_x"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
        result_arrays_double["fulldata_momentum_direction_y"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
        result_arrays_double["fulldata_momentum_direction_z"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
        result_arrays_double["fulldata_sensor_id"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
        result_arrays_int["fulldata_particle_id"] = std::shared_ptr<std::vector<int>>(new std::vector<int>());
        result_arrays_double["fulldata_track_kinetic_energy"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
        result_arrays_double["fulldata_track_total_energy"] = std::shared_ptr<std::vector<double>>(new std::vector<double>());
    }
}

double getTrackMomentum(double pt, bool isgamma) {
    double smearing = (pt / 100.) * (pt / 100.) * 0.04 + 0.01;
    if (!isgamma)
        return pt + pt * smearing * G4INCL::Random::gauss();
    else
        return 0;

}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......



void B4aEventAction::EndOfEventAction(const G4Event *event) {

    std::cout << "END OF RUN ACTION\n";

    if (event->IsAborted()) {//don't fill information
        G4cout << "event was aborted, not writing output" << G4endl;
        clear();
        return;
    }


    G4cout << "nsteps_ " << nsteps_ << G4endl;
    G4cout << "Particles entering counted  " << particles_caught.particles_vertex_position_x.size() << G4endl;
//    G4cout << "Positions tracked  " << positions_tracked_particle_index.size() << G4endl;

    int num_tagged_particles=0;
    for(size_t i = 0; i< particles_caught.particles_vertex_position_x.size();i++) {
        num_tagged_particles += (int)(particles_caught.particles_tagged[i]);
    }
    G4cout << "Particles tagged  " << num_tagged_particles << G4endl;

    float total_true = 0;

    G4cout << "Total true " << total_true << " and total dep " << total_deposit_all << G4endl;
    G4cout << "Hits: "<<rechit_energy_.size() <<G4endl;

}


template<typename T>
void B4aEventAction::writeToBinaryFile(std::string filename, std::vector<T> &x, int id, int num_elements) {
    G4String this_event_folder = output_bin_folder + std::string("/") + std::to_string(id);
    std::ofstream file;
    file.open(this_event_folder + std::string("/") + std::string(filename), std::ios::out | std::ios::binary);

    file.write((char *) &num_elements, sizeof(int));

    for (int i = 0; i < x.size(); i++) {
        if (rechit_energy_[i] > 0) {
            T v = x[i];
            file.write((char *) &v, sizeof(T));
        }
    }

    file.close();
}

const std::vector<double> &B4aEventAction::getRechitEnergy() const {
    return rechit_energy_;
}

const std::vector<double> &B4aEventAction::getRechitX() const {
    return rechit_x_;
}

const std::vector<double> &B4aEventAction::getRechitY() const {
    return rechit_y_;
}

const std::vector<double> &B4aEventAction::getRechitZ() const {
    return rechit_z_;
}

const std::vector<int> &B4aEventAction::getHitsParticles() const {
    return hits_particles_id;
}

const std::vector<double> &B4aEventAction::getHitsDeposits() const {
    return hits_particles_deposits;
}

const std::vector<int> &B4aEventAction::getHitsParticlesSensorIdx() const {
    return hits_particles_sensor_idx;
}

const std::vector<int> &B4aEventAction::getRechitStupidPid() const {
    return rechit_stupid_pid;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
