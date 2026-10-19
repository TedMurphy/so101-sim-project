python3 -c "
import re
path = '/home/ted/so101-sim-project/source/sim_to_real_so101/tasks/cylinder_to_target_env_cfg.py'
txt = open(path).read()
txt = re.sub(r'^        rot=np\.array\(\[.*?\]\)', '        rot=np.array([0.0262, 0.0436, 0.9987, -0.0011])', txt, flags=re.MULTILINE)
open(path, 'w').write(txt)
print('Done')
"

2.0°	[0.0174, 0.0436, 0.9988, -0.0008]
2.5°	[0.0218, 0.0436, 0.9987, -0.0010]
3°	[0.0262, 0.0436, 0.9987, -0.0011]
4°	[0.0349, 0.0436, 0.9983, -0.0015]
5°	[0.0436, 0.0436, 0.9981, -0.0019]
6°	[0.0523, 0.0435, 0.9975, -0.0023]


python3 -c "
import re
path = '/home/ted/so101-sim-project/source/sim_to_real_so101/tasks/cylinder_to_target_env_cfg.py'
txt = open(path).read()
txt = re.sub(r'^        rot=np\.array\(\[.*?\]\)', '        rot=np.array([0.0270, 0.0431, 0.9986, 0.0163])', txt, flags=re.MULTILINE)
open(path, 'w').write(txt)
print('Done')
"

+3° yaw: [0.0250, 0.0443, 0.9983, -0.0272]
-1.0°	[0.0266, 0.0434, 0.9987, 0.0076]
-1.5°	[0.0268, 0.0433, 0.9986, 0.0120]
-2.0°	[0.0270, 0.0431, 0.9986, 0.0163]
-2.5°	[0.0271, 0.0430, 0.9985, 0.0207]
-3° yaw: [0.0273, 0.0429, 0.9983, 0.0250]




python3 -c "
import re
path = '/home/ted/so101-sim-project/source/sim_to_real_so101/tasks/cylinder_to_target_env_cfg.py'
txt = open(path).read()
txt = re.sub(r'^        rot=np\.array\(\[.*?\]\)', '        rot=np.array([0.0267, 0.0257, 0.9992, 0.0167])', txt, flags=re.MULTILINE)
open(path, 'w').write(txt)
print('Done')
"

Forward tilt	Quaternion
4°	[0.0268, 0.0344, 0.9990, 0.0165]
3°	[0.0267, 0.0257, 0.9992, 0.0167]
2°	[0.0265, 0.0169, 0.9994, 0.0169]
5° (current)	[0.0270, 0.0431, 0.9986, 0.0163]

5.5°	[0.0270, 0.0475, 0.9984, 0.0162]
6.0°	[0.0270, 0.0519, 0.9982, 0.0161]
6.5°	[0.0271, 0.0562, 0.9979, 0.0159]
7.0°	[0.0272, 0.0605, 0.9977, 0.0158]


Forward tilt	Quaternion
7.5°	[0.0273, 0.0649, 0.9974, 0.0157]
8.0°	[0.0273, 0.0693, 0.9971, 0.0156]
8.5°	[0.0274, 0.0736, 0.9968, 0.0155]
9.0°	[0.0275, 0.0780, 0.9965, 0.0153]

python3 -c "
import re
path = '/home/ted/so101-sim-project/source/sim_to_real_so101/tasks/cylinder_to_target_env_cfg.py'
txt = open(path).read()
txt = re.sub(r'^        rot=np\.array\(\[.*?\]\)', '        rot=np.array([0.0273, 0.0693, 0.9971, 0.0156])', txt, flags=re.MULTILINE)
open(path, 'w').write(txt)
print('Done')
"


ted@ted-Legion-7-16IAX10:~/IsaacLab$ python3 -c "
import re
path = '/home/ted/so101-sim-project/source/sim_to_real_so101/tasks/cylinder_to_target_env_cfg.py'
txt = open(path).read()
txt = re.sub(r'^        rot=np\.array\(\[.*?\]\)', '        rot=np.array([0.0273, 0.0693, 0.9971, 0.0156])', txt, flags=re.MULTILINE)
open(path, 'w').write(txt)
print('Done')
"
Done


Lateral tilt	Quaternion
3.0° (current)	[0.0273, 0.0693, 0.9971, 0.0156]
3.5°	[0.0317, 0.0692, 0.9970, 0.0153]
4.0°	[0.0360, 0.0691, 0.9968, 0.0150]
4.5°	[0.0404, 0.0690, 0.9967, 0.0147]

python3 -c "
import re
path = '/home/ted/so101-sim-project/source/sim_to_real_so101/tasks/cylinder_to_target_env_cfg.py'
txt = open(path).read()
txt = re.sub(r'^        rot=np\.array\(\[.*?\]\)', '        rot=np.array([0.0273, 0.0693, 0.9971, 0.0156])', txt, flags=re.MULTILINE)
open(path, 'w').write(txt)
print('Done')
"

