def run(outcomes_path, decisions_path):
    # Import necessary components.
    from utils.sim import (blood_type_compatible, hla_match_score, 
        default_matching_strategy, Simulation, 
        TransplantCenter, Hospital)

    # Build a network of hospitals.
    center = TransplantCenter()
    hosp1 = Hospital("General_Hospital")
    hosp2 = Hospital("City_Med")
    hosp3 = Hospital("Regional_Care")
    center.add_hospital(hosp1)
    center.add_hospital(hosp2)
    center.add_hospital(hosp3)

    # Define connections between hospitals (travel time in hours).
    center.add_edge("General_Hospital", "City_Med", travel_time=1.5)
    center.add_edge("City_Med", "Regional_Care", travel_time=2.0)
    center.add_edge("General_Hospital", "Regional_Care", travel_time=3.0)

    # Optionally, plug in a custom matching strategy.
    def custom_strategy(patient, organ):
        if not blood_type_compatible(organ.blood_type, patient.blood_type):
            return 0.0
        if not patient.crossmatch(organ):
            return 0.0
        return 1.0 + 0.3 * patient.wait_time + 1.2 * patient.urgency
    center.matching_strategy = custom_strategy

    # Run the simulation.
    sim = Simulation(center, steps=365 * 1, organ_probability=0.7, patient_probability=0.3)
    sim.run()
    metrics = sim.calculate_metrics()
    sim.save_logs(outcomes_path, decisions_path)

    return metrics

    # # Run a parameter sweep.
    # param_ranges = {
    #     "organ_probability": [0.6, 0.7, 0.8],
    #     "patient_probability": [0.2, 0.3, 0.4]
    # }
    # Simulation.parameter_sweep(param_ranges, steps=365 * 1, runs_per_combination=3)