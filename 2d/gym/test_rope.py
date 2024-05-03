from rope import Rope

rope = Rope()
q0 = [0,  -54,  134, -167,  -90,    0] 
qf = [0,  -88.33333333,   93.33333333, -183.33333333,  -90,    0]


rope.run_sim(q0, qf)
