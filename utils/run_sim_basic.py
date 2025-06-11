from utils.sim import TransplantCenter, Hospital, Simulation

center = TransplantCenter()
hosp1 = Hospital("Hospital A")
hosp2 = Hospital("Hospital B")
center.add_hospital(hosp1)
center.add_hospital(hosp2)

sim = Simulation(center, steps=100)
sim.run()
metrics = sim.calculate_metrics()
sim.save_logs()