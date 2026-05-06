from data.behaviors.corrigibility import pairs as c_pairs
from data.behaviors.power_seeking import pairs as p_pairs

print("Corrigibility example:")
print(repr(c_pairs[0]["positive"][-30:]))

print("Power_seeking example:") 
print(repr(p_pairs[0]["positive"][-30:]))