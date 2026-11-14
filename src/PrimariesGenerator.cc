//
// Created by Shah Rukh Qasim on 1/6/22.
//

#include "PrimariesGenerator.hh"

#include "G4RunManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "G4INCLRandom.hh"
#include <G4INCLGeant4Random.hh>
#include <G4INCLRandomSeedVector.hh>
#include <G4INCLRanecu.hh>
#include<ctime>
#include <cmath>
#include <algorithm>
#include<sys/types.h>
#include <cstdlib>
#include "mixmax2.hh"


void PrimariesGenerator::init_pythia(Pythia8::Pythia *pythia, bool qqbar2ttbar) {
    pythia->readString("Beams:eCM = 14000.");
    pythia->readString("Init:showChangedParticleData = off");
    pythia->readString("Init:showChangedSettings = on");
    pythia->readString("Next:numberShowLHA = 0");
    pythia->readString("Next:numberShowInfo = 0");
    pythia->readString("Next:numberShowProcess = 0");
    pythia->readString("Next:numberShowEvent = 0");

    pythia->readString("Tune:pp 14");
    pythia->readString("Tune:ee 7");
    pythia->readString("MultipartonInteractions:ecmPow=0.03344");
    pythia->readString("MultipartonInteractions:bProfile=2");
    pythia->readString("MultipartonInteractions:pT0Ref=1.41");
    pythia->readString("MultipartonInteractions:coreRadius=0.7634");
    pythia->readString("MultipartonInteractions:coreFraction=0.63");
    pythia->readString("ColourReconnection:range=5.176");
    pythia->readString("SigmaTotal:zeroAXB=off");
    pythia->readString("SpaceShower:alphaSorder=2");
    pythia->readString("SpaceShower:alphaSvalue=0.118");
    pythia->readString("SigmaProcess:alphaSvalue=0.118");
    pythia->readString("SigmaProcess:alphaSorder=2");
    pythia->readString("MultipartonInteractions:alphaSvalue=0.118");
    pythia->readString("MultipartonInteractions:alphaSorder=2");
    pythia->readString("TimeShower:alphaSorder=2");
    pythia->readString("TimeShower:alphaSvalue=0.118");
    pythia->readString("SigmaTotal:mode = 0");
    pythia->readString("SigmaTotal:sigmaEl = 21.89");
    pythia->readString("SigmaTotal:sigmaTot = 100.309");
    //pythia->readString("PDF:pSet=LHAPDF6:NNPDF31_nnlo_as_0118");

    //  pythia->readString("SoftQCD:nonDiffractive = on");
    //  pythia->readString("SoftQCD:singleDiffractive = on");
    //  pythia->readString("SoftQCD:doubleDiffractive = on");

    if(qqbar2ttbar)
        pythia->readString("Top:qqbar2ttbar = on");
    else
        pythia->readString("SoftQCD:inelastic = on");

    pythia->readString("Tune:preferLHAPDF = 2");
    pythia->readString("Main:timesAllowErrors = 10000");
    pythia->readString("Check:epTolErr = 0.01");
    pythia->readString("Beams:setProductionScalesFromLHEF = off");
    pythia->readString("SLHA:minMassSM = 1000.");
    pythia->readString("ParticleDecays:limitTau0 = on");
    pythia->readString("ParticleDecays:tau0Max = 10");
    pythia->readString("ParticleDecays:allowPhotonRadiation = on");

    pythia->init();
    std::cout << "Pythia initialized..." << std::endl;
}

PrimariesGenerator::PrimariesGenerator(std::string pythiadata, int seed_0, int seed_1, int seed_2, int seed_3) :
        pythia_("/Users/shahrukhqasim/Workspace/NextCal/miniCalo/pythia8-data"),
        pythia_qqbar2ttbar("/Users/shahrukhqasim/Workspace/NextCal/miniCalo/pythia8-data"),
        jetDef_(new fastjet::JetDefinition(fastjet::antikt_algorithm, 0.4)) {


    G4INCL::Random::setGenerator(new G4INCL::Ranecu());

    init_pythia(&pythia_, false);
    init_pythia(&pythia_qqbar2ttbar, true);

    random_engine = new MixMaxRndm(seed_0, seed_1, seed_2, seed_3);
    MixMaxRndm*casted_rand = (MixMaxRndm*)random_engine;

    pythia_.rndm.rndmEnginePtr(random_engine);
    pythia_qqbar2ttbar.rndm.rndmEnginePtr(random_engine);
//    pythia_.rndm.init(seed_0);
//    pythia_qqbar2ttbar.rndm.init(seed_0);

    fParticleGun = new G4ParticleGun();
}


void PrimariesGenerator::generate() {
}


void PrimariesGenerator::GenerateSingleVertex(G4PrimaryVertex *vertex) {
    double const etaTargetMin = 1.5;
    double const etaTargetMax = 3.0;
    double const eMin = 10.;
    double const eMax = 4000.;

    // jets from this position at eta ~ 3.6 will hit the center of the detector
    xorig_ = 0;
    yorig_ = 0;

    G4double zsign = 1.;
    std::vector<int> primaries;
    G4ThreeVector vertex_position(xorig_, yorig_, 0);
    G4double vertex_time(0.);

    if(not (simulation_type==SimType::pu or simulation_type==SimType::qqbar2ttbar))
        throw std::runtime_error("Error occurred in pythia type. Please check.");

    Pythia8::Pythia* pythia_in_use = simulation_type==SimType::pu?&pythia_:&pythia_qqbar2ttbar;

//    std::cout<<"Gen sim type "<<getSimTypeAsString()<<std::endl;

    bool success;
    for (int i = 0 ; i< 1000;i++) {
        success = pythia_in_use->next();
        success = success and pythia_in_use->event.size() > 0;
        if (success)
            break;
    }

    if (not success)
        throw std::runtime_error("Pythia failed to generate event after 1000 tries.");

    primaries.clear();
    double tota_energy = 0;

    for (int i=0; i < pythia_in_use->event.size(); ++i) {
        auto &part(pythia_in_use->event[i]);

        if (part.isFinal()) {
            if (part.eta() < eta_max_lim && part.eta() > eta_min_lim) {
//                    std::cout<<"B eta "<<part.eta()<<" and e "<<part.e()<<std::endl;

                primaries.push_back(i);
                tota_energy += part.e();
            }
        }
    }
//    std::cout << "Total particles " << pythia_in_use->event.size()<<std::endl;
//    std::cout<< "Particles 1 < eta < 4 " << primaries.size()<<std::endl;
//    std::cout << "Total energy 1 < eta < 4 " << tota_energy << std::endl;


    for (int ipart: primaries) {
        auto &pj(pythia_in_use->event[ipart]);

        int pdgId = pj.id();
        auto *partDefinition = G4ParticleTable::GetParticleTable()->FindParticle(pdgId);
        if (partDefinition == nullptr)
            continue; //throw std::runtime_error(std::string("Unknown particle ") + std::to_string(pdgId));

        auto *particle = new G4PrimaryParticle(pdgId);
        particle->SetMass(pj.m() * GeV);
        particle->SetMomentum(pj.px() * GeV, pj.py() * GeV, pj.pz() * GeV * zsign);
        particle->SetCharge(partDefinition->GetPDGCharge());
        vertex->SetPrimary(particle);
    }
}

bool PrimariesGenerator::isJetGenerator() {
    return false;
}

std::vector<G4String> PrimariesGenerator::generateAvailableParticles() const {
    return {"isMinbias"};
}

B4PartGeneratorBase::particles PrimariesGenerator::getParticle() const {
    return minbias;
}

int PrimariesGenerator::isParticle(int i) const {
    return 0;
}

void PrimariesGenerator::GenerateParticle(G4Event *anEvent) {
    G4double worldZHalfLength = 0.;
    auto worldLV = G4LogicalVolumeStore::GetInstance()->GetVolume("World");

    // Check that the world volume has box shape
    G4Box *worldBox = nullptr;
    if (worldLV) {
        worldBox = dynamic_cast<G4Box *>(worldLV->GetSolid());
    }

    if (worldBox) {
        worldZHalfLength = worldBox->GetZHalfLength();
    } else {
        G4ExceptionDescription msg;
        msg << "World volume of box shape not found." << G4endl;
        msg << "Perhaps you have changed geometry." << G4endl;
        msg << "The gun will be place in the center.";
        G4Exception("B4PrimaryGeneratorAction::GeneratePrimaries()",
                    "MyCode0002", JustWarning, msg);
    }
    // Set gun position
    int id = anEvent->GetEventID();
    G4ThreeVector vertex_position(xorig_, yorig_, 0);
    G4double vertex_time(0.);
    G4PrimaryVertex *vertex = new G4PrimaryVertex(vertex_position, vertex_time);

    G4ThreeVector position(particle_position_x * m, particle_position_y * m, particle_position_z * m);
    G4ThreeVector direction(particle_direction_x * m, particle_direction_y * m, particle_direction_z * m);

    auto *partDefinition = G4ParticleTable::GetParticleTable()->FindParticle(particle_pdgid);
    if (partDefinition == nullptr)
        throw std::runtime_error(std::string("Unknown particle ") + std::to_string(particle_pdgid));

    std::cout<<"Shooting particle with energy "<<particle_energy<<" GeV"<<std::endl;
    std::cout<<"Position "<<position.x()<<" "<<position.y()<<" "<<position.z()<<" "<<std::endl;
    std::cout<<"Momentum direction "<<direction.x()<<" "<<direction.y()<<" "<<direction.z()<<" "<<std::endl;
    std::cout<<"PDGID "<<particle_pdgid<<std::endl;

    fParticleGun->SetParticleDefinition(partDefinition);
    fParticleGun->SetParticleMomentumDirection(direction);
    fParticleGun->SetParticleEnergy(particle_energy * GeV);
    fParticleGun->SetParticlePosition(position);
    {   // HIT TIME: a particle created away from the origin (singlepart_face) starts at its time of flight from the
        // origin, so that every simulation's Geant4 global time counts from the collision (0 for the origin).
        G4double mass = partDefinition->GetPDGMass();
        G4double etot = particle_energy * GeV + mass;
        G4double beta = std::sqrt(std::max(0.0, 1.0 - (mass / etot) * (mass / etot)));
        fParticleGun->SetParticleTime(beta > 0 ? position.mag() / (beta * CLHEP::c_light) : 0.0);
    }
    fParticleGun->GeneratePrimaryVertex(anEvent);
}
void PrimariesGenerator::GenerateWithPythia(G4Event *anEvent) {
    G4ThreeVector vertex_position(xorig_, yorig_, 0);
    G4double vertex_time(0.);
    G4PrimaryVertex *vertex = new G4PrimaryVertex(vertex_position, vertex_time);
    GenerateSingleVertex(vertex);

    anEvent->AddPrimaryVertex(vertex);
}

G4PrimaryVertex * PrimariesGenerator::GeneratePUParticles() {
    eta_min_lim = -100000;
    eta_max_lim = 1000000;

    G4ThreeVector vertex_position(xorig_, yorig_, 0);
    G4double vertex_time(0.);
    G4PrimaryVertex *vertex = new G4PrimaryVertex(vertex_position, vertex_time);
    GenerateSingleVertex(vertex);
    return vertex;
}


void PrimariesGenerator::GeneratePrimaries(G4Event *anEvent) {
    switch (simulation_type) {
        case SimType::pu:
        case SimType::qqbar2ttbar:
            GenerateWithPythia(anEvent);
            break;
        case SimType::singlepart:
            GenerateParticle(anEvent);
            break;
    }
}


void PrimariesGenerator::SetNextToPU() {
    simulation_type=SimType::pu;
    eta_min_lim = 1;
    eta_max_lim = 4;
}

void PrimariesGenerator::SetNextTo_qqbar_to_ttbar() {
    simulation_type=SimType::qqbar2ttbar;
}

void PrimariesGenerator::SetNextToParticle(double position_x, double position_y, double position_z, double direction_x,
                                           double direction_y,
                                           double direction_z, int pdgid, double energy) {
    simulation_type=SimType::singlepart;

    particle_position_x = position_x;
    particle_position_y = position_y;
    particle_position_z = position_z;

    particle_direction_x = direction_x;
    particle_direction_y = direction_y;
    particle_direction_z = direction_z;

    particle_pdgid = pdgid;
    particle_energy = energy;
}

std::string PrimariesGenerator::getSimTypeAsString() {
    std::string simtype_string;
    switch (simulation_type) {
        case SimType::pu:
            simtype_string="minbias";
            break;
        case SimType::qqbar2ttbar:
            simtype_string="qqbar2ttbar";
            break;
        case SimType::singlepart:
            simtype_string="singlepart";
            break;
        default:
            simtype_string="unknown";
    }
    return simtype_string;
}