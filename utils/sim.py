"""
Kidney Transplant Allocation Simulation Framework

This module provides a comprehensive simulation framework for modeling and evaluating 
kidney transplantation allocation policies across a network of hospitals.

Key Features:
- Stochastic simulation of organ donation and patient matching
- Detailed modeling of donor, organ, and patient characteristics
- Flexible matching strategies with customizable parameters
- Network-level simulation supporting multiple hospitals
- Comprehensive metrics calculation and logging

Core Components:
1. Domain Models:
   - Donor: Represents organ donors with detailed attributes
   - Organ: Represents individual kidney organs with quality metrics
   - Patient: Represents transplant candidates with compatibility factors

2. Network Simulation:
   - Hospital: Local matching and transplant simulation
   - TransplantCenter: Manages inter-hospital organ exchanges

3. Simulation Engine:
   - Supports probabilistic patient and organ arrivals
   - Parallel simulation capabilities
   - Detailed outcome tracking and metrics calculation

Usage Example:
    center = TransplantCenter()
    hosp1 = Hospital("Hospital A")
    hosp2 = Hospital("Hospital B")
    center.add_hospital(hosp1)
    center.add_hospital(hosp2)
    
    sim = Simulation(center, steps=100)
    sim.run()
    metrics = sim.calculate_metrics()
    sim.save_logs()

Authors: Aniruth Ananthanarayanan and Benjamin Zijan Hu
Version: 1.0.0
Last Updated: 2025-04-05
"""

import random
import csv
from functools import wraps
from itertools import product
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------
# Utility Functions & Decorators
# -------------------------------------

def log_call(func):
    """
    Decorator to log when a function is called and finished.
    
    Args:
        func: The function to be logged.
    
    Returns:
        A wrapper function that logs the execution of the input function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        #print(f"[LOG] Executing {func.__name__}...")
        result = func(*args, **kwargs)
        #print(f"[LOG] Finished {func.__name__}.\n")
        return result
    return wrapper

def blood_type_compatible(donor_bt, patient_bt):
    """
    Return True if the donor's blood type is compatible with the patient.
    
    Args:
        donor_bt (str): The donor's blood type.
        patient_bt (str): The patient's blood type.
    
    Returns:
        bool: Whether the donor's blood type is compatible with the patient.
    """
    compatibility_chart = {
        'O': ['O', 'A', 'B', 'AB'],
        'A': ['A', 'AB'],
        'B': ['B', 'AB'],
        'AB': ['AB']
    }
    return patient_bt in compatibility_chart.get(donor_bt, [])

def hla_match_score(patient_hla, donor_hla):
    """
    Return a matching score (0 to 1) based on overlapping HLA antigens.
    
    Args:
        patient_hla (list): The patient's HLA antigens.
        donor_hla (list): The donor's HLA antigens.
    
    Returns:
        float: The HLA matching score.
    """
    if not donor_hla:
        return 0.0
    matches = len(set(patient_hla).intersection(set(donor_hla)))
    return matches / len(donor_hla)

def default_matching_strategy(patient, organ):
    """
    Default matching strategy incorporating:
      - Blood type compatibility
      - Simulated crossmatch test
      - HLA matching
      - Wait time & urgency
      - Organ quality
      - Donor–patient body size compatibility
    
    Args:
        patient (Patient): The patient to be matched.
        organ (Organ): The organ to be matched.
    
    Returns:
        float: The matching score.
    """
    if not blood_type_compatible(organ.blood_type, patient.blood_type):
        return 0.0
    if not patient.crossmatch(organ):
        return 0.0
    hla_score = hla_match_score(patient.hla, organ.donor.hla)
    size_factor = 1 - abs(patient.body_size - organ.donor.body_size) / max(patient.body_size, organ.donor.body_size)
    score = (2.0 +                  # Base bonus for compatible blood type
             5.0 * hla_score +       # HLA matching bonus
             0.1 * patient.wait_time +  # Bonus for longer wait
             0.5 * patient.urgency +    # Bonus for urgency
             2.0 * organ.kidney_quality)  # Bonus for organ quality
    return score * size_factor

# -------------------------------------
# Domain Classes: Donor, Organ, Patient
# -------------------------------------

class Donor:
    """
    Represents an organ donor.
    
    Attributes:
        donation_type (str): "living" or "deceased"
        donation_subtype (str): e.g., for living ("related", "paired_exchange") or for deceased ("SCD", "ECD", "DCD")
    """
    def __init__(self, donor_id, blood_type, age, hla, health_score, body_size, donation_type, donation_subtype=None):
        """
        Initializes a Donor instance.
        
        Args:
            donor_id (str): The donor's ID.
            blood_type (str): The donor's blood type.
            age (int): The donor's age.
            hla (list): The donor's HLA antigens.
            health_score (float): The donor's health score (0 to 1).
            body_size (float): The donor's body size (e.g., in kilograms).
            donation_type (str): The donation type ("living" or "deceased").
            donation_subtype (str, optional): The donation subtype. Defaults to None.
        """
        self.id = donor_id
        self.blood_type = blood_type
        self.age = age
        self.hla = hla
        self.health_score = health_score  # 0 to 1 (1 is optimal)
        self.body_size = body_size        # e.g., in kilograms
        self.donation_type = donation_type
        self.donation_subtype = donation_subtype

class Organ:
    """
    Represents a kidney from a donor.
    
    Includes details for preservation and transportation.
    """
    def __init__(self, organ_id, donor: Donor):
        """
        Initializes an Organ instance.
        
        Args:
            organ_id (str): The organ's ID.
            donor (Donor): The donor of the organ.
        """
        self.id = organ_id
        self.donor = donor
        self.blood_type = donor.blood_type
        # Procurement & transport details:
        self.preservation_method = random.choice(["cold_storage", "machine_perfusion"])
        self.preservation_time = random.uniform(0, 36)  # Hours in preservation
        self.transportation_method = random.choice(["air", "ground"])
        self.transportation_delay = random.uniform(0, 12)  # Hours delay
        # Calculate kidney quality and expected lifespan.
        self.kidney_quality = self.calculate_kidney_quality()
        self.expected_lifespan = self.calculate_expected_lifespan()

    def calculate_kidney_quality(self):
        """
        Calculates the kidney quality based on the donor's health score and age.
        
        Returns:
            float: The kidney quality (0 to 1).
        """
        base_quality = self.donor.health_score
        age_penalty = max(0, (self.donor.age - 30) / 100)
        preservation_penalty = max(0, (self.preservation_time - 24) / 24) * 0.2
        quality = max(0, base_quality - age_penalty - preservation_penalty)
        return round(quality, 2)

    def calculate_expected_lifespan(self):
        """
        Calculates the expected lifespan of the kidney based on the donor's age and kidney quality.
        
        Returns:
            float: The expected lifespan (years).
        """
        base_lifespan = 15  # years
        age_factor = max(0, (70 - self.donor.age) / 40)
        expected = base_lifespan * self.kidney_quality * age_factor
        delay_factor = max(0.5, 1 - self.transportation_delay / 24)
        expected *= delay_factor
        return round(expected, 1)

class Patient:
    """
    Represents a patient awaiting a kidney transplant.
    
    Attributes:
        blood_type (str): The patient's blood type.
        hla (list): The patient's HLA antigens.
        urgency (int): The patient's urgency (1 to 5).
        health_status (float): The patient's health status (0 to 1).
        body_size (float): The patient's body size (e.g., in kilograms).
        pra (float): The patient's Panel Reactive Antibody (0 to 1).
    """
    def __init__(self, patient_id, blood_type, hla, urgency, health_status, body_size, pra):
        """
        Initializes a Patient instance.
        
        Args:
            patient_id (str): The patient's ID.
            blood_type (str): The patient's blood type.
            hla (list): The patient's HLA antigens.
            urgency (int): The patient's urgency (1 to 5).
            health_status (float): The patient's health status (0 to 1).
            body_size (float): The patient's body size (e.g., in kilograms).
            pra (float): The patient's Panel Reactive Antibody (0 to 1).
        """
        self.id = patient_id
        self.blood_type = blood_type
        self.hla = hla
        self.urgency = urgency          # Scale 1-5
        self.wait_time = 0              # In days
        self.health_status = health_status
        self.body_size = body_size
        self.pra = pra                  # Panel Reactive Antibody (0 to 1)
        self._compatibility_cache = {}
        self._dirty = True

    def update_wait_time(self, days=1):
        """
        Updates the patient's wait time.
        
        Args:
            days (int, optional): The number of days to add to the wait time. Defaults to 1.
        """
        self.wait_time += days
        self._dirty = True

    def crossmatch(self, organ: Organ):
        """
        Simulates a crossmatch test; higher PRA lowers chance to pass.
        
        Args:
            organ (Organ): The organ to be crossmatched.
        
        Returns:
            bool: Whether the crossmatch test passes.
        """
        return random.random() > self.pra

    def compute_default_compatibility(self, organ: Organ):
        """
        Computes the default compatibility score between the patient and the organ.
        
        Args:
            organ (Organ): The organ to be matched.
        
        Returns:
            float: The compatibility score.
        """
        if self._dirty:
            self._compatibility_cache.clear()
            self._dirty = False
        if organ.id in self._compatibility_cache:
            return self._compatibility_cache[organ.id]
        score = default_matching_strategy(self, organ)
        self._compatibility_cache[organ.id] = score
        return score

# -------------------------------------
# Hospital & Network Classes
# -------------------------------------

class Hospital:
    """
    Represents a hospital node.
    
    Maintains its own waiting list (patients) and organ inventory.
    Logs matching decisions and transplant outcomes.
    """
    def __init__(self, name):
        """
        Initializes a Hospital instance.
        
        Args:
            name (str): The hospital's name.
        """
        self.name = name
        self.patients = []     # List[Patient]
        self.organs = []       # List[Organ]
        self.matching_decisions = []  # Log of matching decisions
        self.transplant_outcomes = [] # Log of transplant outcomes

    def add_patient(self, patient: Patient):
        """
        Adds a patient to the hospital's waiting list.
        
        Args:
            patient (Patient): The patient to be added.
        """
        self.patients.append(patient)

    def add_organ(self, organ: Organ):
        """
        Adds an organ to the hospital's inventory.
        
        Args:
            organ (Organ): The organ to be added.
        """
        self.organs.append(organ)

    @log_call
    def perform_matching(self, donor_hospital: 'Hospital', matching_strategy=None, extra_penalty=0.0, ):
        """
        Matches available organs to patients.
        An extra penalty (e.g., for inter-hospital transfers) can be applied.
        Returns a dict mapping organ.id to patient.id.
        
        Args:
            matching_strategy (function, optional): A custom matching strategy. Defaults to None.
            extra_penalty (float, optional): An extra penalty to be applied. Defaults to 0.0.
        
        Returns:
            dict: A dictionary mapping organ IDs to patient IDs.
        """

        if (donor_hospital == None):
            donor_hospital = self

        # print(self.name, "performing matching with donor hospital", donor_hospital.name)

        matches = {}
        for organ in donor_hospital.organs:
            best_score = -1.0
            best_patient = None
            for patient in self.patients:
                score = matching_strategy(patient, organ) if matching_strategy else patient.compute_default_compatibility(organ)
                score *= (1 - extra_penalty)  # apply travel or other penalties if needed
                if score > best_score:
                    best_score = score
                    best_patient = patient
            if best_patient and best_score > 0:
                matches[organ.id] = best_patient.id
                decision = {
                    "hospital": self.name,
                    "donor_hospital": donor_hospital.name,
                    "organ_id": organ.id,
                    "donor_id": organ.donor.id,
                    "donor_type": organ.donor.donation_type,
                    "donor_subtype": organ.donor.donation_subtype,
                    "patient_id": best_patient.id,
                    "matching_score": round(best_score, 2),
                    "organ_quality": organ.kidney_quality,
                    "expected_lifespan": organ.expected_lifespan,
                    "patient_wait_time": best_patient.wait_time
                }
                self.matching_decisions.append(decision)
                # Simulate transplant outcome.
                self.simulate_transplant(best_patient, organ, best_score)
                # Remove matched patient.
                self.patients = [p for p in self.patients if p.id != best_patient.id]
        self.organs = []  # Clear organs once processed.
        return matches

    def simulate_transplant(self, patient, organ, score):
        """
        Simulates transplant outcome.
        Higher matching scores increase probability of success.
        Records additional metrics.
        
        Args:
            patient (Patient): The patient who received the transplant.
            organ (Organ): The organ that was transplanted.
            score (float): The matching score.
        """
        success_prob = min(1, score / 20.0)
        outcome = "Success" if random.random() < success_prob else "Rejection"
        record = {
            "hospital": self.name,
            "organ_id": organ.id,
            "patient_id": patient.id,
            "matching_score": round(score, 2),
            "transplant_outcome": outcome,
            "patient_wait_time": patient.wait_time,
            "expected_lifespan": organ.expected_lifespan,
            "donor_type": organ.donor.donation_type,
            "donor_subtype": organ.donor.donation_subtype,
            "donor_age": organ.donor.age,
            "organ_quality": organ.kidney_quality,
            "patient_urgency": patient.urgency
        }
        self.transplant_outcomes.append(record)
        #print(f"[{self.name}] Transplant Outcome: Patient {patient.id} with Organ {organ.id} -> {outcome}")

class TransplantCenter:
    """
    Represents a network of hospitals.
    
    Nodes are hospitals; edges represent inter-hospital travel (with travel time).
    Coordinates local and inter-hospital matching.
    """
    def __init__(self):
        """
        Initializes a TransplantCenter instance.
        """
        self.hospitals = {}  # hospital name -> Hospital instance
        self.edges = {}      # (hosp1, hosp2) -> travel_time (hours)
        self.matching_strategy = None  # Optional custom matching strategy

    def add_hospital(self, hospital: Hospital):
        """
        Adds a hospital to the network.
        
        Args:
            hospital (Hospital): The hospital to be added.
        """
        self.hospitals[hospital.name] = hospital

    def add_edge(self, hosp1_name, hosp2_name, travel_time):
        """
        Adds a bidirectional connection (edge) between two hospitals.
        
        Args:
            hosp1_name (str): The name of the first hospital.
            hosp2_name (str): The name of the second hospital.
            travel_time (float): The travel time between the two hospitals.
        """
        self.edges[(hosp1_name, hosp2_name)] = travel_time
        self.edges[(hosp2_name, hosp1_name)] = travel_time

    def get_neighbors(self, hospital_name):
        """
        Returns a list of (neighbor_name, travel_time) for a given hospital.
        
        Args:
            hospital_name (str): The name of the hospital.
        
        Returns:
            list: A list of tuples containing the names of neighboring hospitals and their travel times.
        """
        neighbors = []
        for (h1, h2), t in self.edges.items():
            if h1 == hospital_name:
                neighbors.append((h2, t))
        return neighbors

    # TODO - Rightnow this just matches everything in local hopsitals; implement inter-hospital matching as well as 
    # method of passing a better matching strategy

    @log_call
    def perform_network_matching(self):
        """
        Performs local matching in each hospital.
        (Inter-hospital matching can be added by adjusting extra penalties based on travel time.)
        Returns a dictionary mapping organ IDs to patient IDs.
        
        Returns:
            dict: A dictionary mapping organ IDs to patient IDs.
        """
        network_matches = {}
        for hosp in self.hospitals.values():
            local_matches = hosp.perform_matching(matching_strategy=self.matching_strategy, donor_hospital=hosp)
            hosp_neighbors = self.get_neighbors(hosp.name)
            for neighbor_name, travel_time in hosp_neighbors:
                # Apply extra penalty based on travel time (e.g., 10% per hour).
                extra_penalty = 0.1 * travel_time / 24  # Convert hours to days
                neighbor_hospital = self.hospitals[neighbor_name]
                if (hosp != neighbor_hospital): 
                    neighbor_matches = hosp.perform_matching(
                        matching_strategy=self.matching_strategy,
                        extra_penalty=extra_penalty,
                        donor_hospital=neighbor_hospital
                    )
            # network_matches.update({f"{hosp.name}:{org_id}": patient_id for org_id, patient_id in local_matches.items()})
        return network_matches

# -------------------------------------
# Simulation & Metrics Calculation
# -------------------------------------

class Simulation:
    """
    Simulation framework for the kidney transplantation pipeline over a hospital network.
    
    API:
      - simulate_step(): Process one time step (update wait times, add new organs/patients, perform matching).
      - run(): Run simulation for a number of steps.
      - save_logs(): Save outcomes (CSV) and matching decisions (TXT).
      - calculate_metrics(): Compute many evaluation metrics.
      - parameter_sweep(): Run simulations over a grid of parameters.
    """
    def __init__(self, center: TransplantCenter, steps=10, organ_probability=0.7, patient_probability=0.3, seed=None):
        """
        Initializes a Simulation instance.
        
        Args:
            center (TransplantCenter): The transplant center.
            steps (int, optional): The number of simulation steps. Defaults to 10.
            organ_probability (float, optional): The probability of organ arrival. Defaults to 0.7.
            patient_probability (float, optional): The probability of patient arrival. Defaults to 0.3.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        if seed is not None:
            random.seed(seed)
        self.center = center
        self.current_time = 0
        self.steps = steps
        self.organ_probability = organ_probability
        self.patient_probability = patient_probability

    def simulate_step(self):
        """
        Processes one time step (update wait times, add new organs/patients, perform matching).
        """
        #print(f"\n=== Time Step {self.current_time} ===")
        # Update wait times for all patients in every hospital.
        for hosp in self.center.hospitals.values():
            for patient in hosp.patients:
                patient.update_wait_time(days=1)
        # For each hospital, randomly add new organs and patients.
        for hosp in self.center.hospitals.values():
            # Organ arrival.
            if random.random() < self.organ_probability:
                blood_types = ['O', 'A', 'B', 'AB']
                donation_type = random.choice(["living", "deceased"])
                donation_subtype = (random.choice(["related", "unrelated", "paired_exchange", "domino"])
                                    if donation_type == "living" else
                                    random.choice(["SCD", "ECD", "DCD"]))
                donor = Donor(
                    donor_id=f"D{self.current_time}_{random.randint(100,999)}",
                    blood_type=random.choice(blood_types),
                    age=random.randint(18, 70),
                    hla=random.sample(['A1', 'A2', 'B7', 'B8', 'DR3', 'DR4', 'DR15'], 3),
                    health_score=round(random.uniform(0.5, 1.0), 2),
                    body_size=random.uniform(50, 100),
                    donation_type=donation_type,
                    donation_subtype=donation_subtype
                )
                organ = Organ(
                    organ_id=f"O{self.current_time}_{random.randint(100,999)}",
                    donor=donor
                )
                hosp.add_organ(organ)
                #print(f"[{hosp.name}] New Organ: {organ.id} | {donor.donation_type}({donor.donation_subtype}), Blood: {organ.blood_type}, Quality: {organ.kidney_quality:.2f}, Exp. Life: {organ.expected_lifespan} yrs")
            # Patient arrival.
            if random.random() < self.patient_probability:
                blood_types = ['O', 'A', 'B', 'AB']
                patient = Patient(
                    patient_id=f"P{self.current_time}_{random.randint(100,999)}",
                    blood_type=random.choice(blood_types),
                    hla=random.sample(['A1', 'A2', 'B7', 'B8', 'DR3', 'DR4', 'DR15'], 3),
                    urgency=random.randint(1, 5),
                    health_status=round(random.uniform(0.5, 1.0), 2),
                    body_size=random.uniform(50, 100),
                    pra=round(random.uniform(0, 1), 2)
                )
                hosp.add_patient(patient)
                #print(f"[{hosp.name}] New Patient: {patient.id} | Blood: {patient.blood_type}, Urgency: {patient.urgency}, PRA: {patient.pra}")
        # Perform matching across the network.
        self.center.perform_network_matching()
        self.current_time += 1

    def run(self, steps=None):
        """
        Runs the simulation for a number of steps.
        
        Args:
            steps (int, optional): The number of simulation steps. Defaults to None.
        """
        steps = steps if steps is not None else self.steps
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.simulate_step) for _ in range(steps)]
            for future in futures:
                future.result()

    def save_logs(self, outcomes_csv="simulation_outcomes.csv", decisions_txt="matching_decisions.txt"):
        """
        Saves outcomes (CSV) and matching decisions (TXT).
        
        Args:
            outcomes_csv (str, optional): The file name for the outcomes CSV. Defaults to "simulation_outcomes.csv".
            decisions_txt (str, optional): The file name for the matching decisions TXT. Defaults to "matching_decisions.txt".
        """
        # Collect outcomes and decisions from all hospitals.
        all_outcomes = []
        all_decisions = []
        for hosp in self.center.hospitals.values():
            all_outcomes.extend(hosp.transplant_outcomes)
            all_decisions.extend(hosp.matching_decisions)
        if all_outcomes:
            with open(outcomes_csv, "w", newline="") as csvfile:
                fieldnames = all_outcomes[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for rec in all_outcomes:
                    writer.writerow(rec)
            print(f"Transplant outcomes saved to {outcomes_csv}.")
        else:
            print("No transplant outcomes to save.")
        if all_decisions:
            with open(decisions_txt, "w") as txtfile:
                for rec in all_decisions:
                    txtfile.write(str(rec) + "\n")
            print(f"Matching decisions saved to {decisions_txt}.")
        else:
            print("No matching decisions to save.")

    def calculate_metrics(self):
        """
        Calculates detailed evaluation metrics across all hospitals.
        
        Metrics include:
          - Total transplants, successes, rejections, and success rate.
          - Average wait time at transplant.
          - Average matching score.
          - Average expected kidney lifespan.
          - Average organ quality.
          - Breakdown by donor type (living vs deceased).
          - Average donor age and patient urgency.
        
        Returns:
            dict: A dictionary of metrics.
        """
        outcomes = []
        total_patients = 0
        for hosp in self.center.hospitals.values():
            outcomes.extend(hosp.transplant_outcomes)
            total_patients += len(hosp.patients)
        if not outcomes:
            print("No outcomes recorded. Cannot calculate metrics.")
            return {}

        total = len(outcomes)
        successes = sum(1 for rec in outcomes if rec["transplant_outcome"] == "Success")
        rejections = total - successes
        success_rate = successes / total

        avg_wait = sum(rec["patient_wait_time"] for rec in outcomes) / total
        avg_score = sum(rec["matching_score"] for rec in outcomes) / total
        avg_expected_life = sum(rec["expected_lifespan"] for rec in outcomes) / total
        avg_organ_quality = sum(rec["organ_quality"] for rec in outcomes) / total
        avg_donor_age = sum(rec["donor_age"] for rec in outcomes) / total
        avg_patient_urgency = sum(rec["patient_urgency"] for rec in outcomes) / total

        # Breakdown by donor type.
        donor_breakdown = {}
        for rec in outcomes:
            donor_type = rec["donor_type"]
            donor_breakdown.setdefault(donor_type, {"count": 0, "successes": 0})
            donor_breakdown[donor_type]["count"] += 1
            if rec["transplant_outcome"] == "Success":
                donor_breakdown[donor_type]["successes"] += 1

        for dt in donor_breakdown:
            donor_breakdown[dt]["success_rate"] = donor_breakdown[dt]["successes"] / donor_breakdown[dt]["count"]

        metrics = {
            "total_transplants": total,
            "total_patients": total_patients,
            "successes": successes,
            "rejections": rejections,
            "overall_success_rate": round(success_rate, 3),
            "average_wait_time": round(avg_wait, 2),
            "average_matching_score": round(avg_score, 2),
            "average_expected_lifespan": round(avg_expected_life, 2),
            "average_organ_quality": round(avg_organ_quality, 2),
            "average_donor_age": round(avg_donor_age, 2),
            "average_patient_urgency": round(avg_patient_urgency, 2),
            "donor_breakdown": donor_breakdown
        }
        print("\n=== Evaluation Metrics ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        return metrics

    @staticmethod
    def parameter_sweep(param_ranges, steps=10, runs_per_combination=1, output_csv="parameter_sweep_results.csv"):
        """
        Runs the simulation over all combinations of parameters.
        
        Args:
            param_ranges (dict): A dictionary mapping parameter names to lists of values.
            steps (int, optional): The number of simulation steps. Defaults to 10.
            runs_per_combination (int, optional): The number of runs per combination. Defaults to 1.
            output_csv (str, optional): The file name for the output CSV. Defaults to "parameter_sweep_results.csv".
        """
        keys = list(param_ranges.keys())
        combinations = list(product(*(param_ranges[key] for key in keys)))
        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for combo in combinations:
                params = dict(zip(keys, combo))
                futures.append(executor.submit(Simulation._run_combination, params, steps, runs_per_combination))
            for future in futures:
                results.extend(future.result())
        if results:
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
            print(f"Parameter sweep results saved to {output_csv}.")
        else:
            print("No results from parameter sweep.")

    @staticmethod
    def _run_combination(params, steps, runs_per_combination):
        """
        Runs a combination of parameters.
        
        Args:
            params (dict): A dictionary of parameters.
            steps (int): The number of simulation steps.
            runs_per_combination (int): The number of runs per combination.
        
        Returns:
            list: A list of results.
        """
        aggregate_metrics = {"total_success": 0, "total_transplants": 0, "avg_wait_time": 0.0}
        for run in range(runs_per_combination):
            # Build a simple network with two hospitals.
            center = TransplantCenter()
            hospA = Hospital("Hospital_A")
            hospB = Hospital("Hospital_B")
            center.add_hospital(hospA)
            center.add_hospital(hospB)
            center.add_edge("Hospital_A", "Hospital_B", travel_time=2)  # 2 hours travel.
            sim = Simulation(center, steps=steps,
                             organ_probability=params.get("organ_probability", 0.7),
                             patient_probability=params.get("patient_probability", 0.3))
            sim.run(steps)
            metrics = sim.calculate_metrics()
            aggregate_metrics["total_success"] += metrics.get("successes", 0)
            aggregate_metrics["total_transplants"] += metrics.get("total_transplants", 0)
            aggregate_metrics["avg_wait_time"] += metrics.get("average_wait_time", 0)
        for key in aggregate_metrics:
            aggregate_metrics[key] /= runs_per_combination
        result_entry = {**params, **aggregate_metrics}
        return [result_entry]