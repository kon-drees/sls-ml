from sls_ml.af_parser import parse_apx
from sls_ml.walkaaf import walkaaf_optimized, is_stable, walkaaf
import networkx as nx

# Given
G = parse_apx("/Users/konraddrees/Documents/GitHub/sls-ml/files/icmma/traffic1.apx")

# When
# Takes time lol
stable_extension = None
count = 0
print("Start")
while(stable_extension == None):
    stable_extension = walkaaf(G, 12*G.number_of_nodes());
# stable_extension = walkaaf_probability_g(G, 0.8, max_flips=2*G.number_of_nodes())
    count += 1
# Then
# Two cases because there exists one stable extension but the algorithm can just find one
    if stable_extension:
        print(count)
        print(stable_extension)
        break
    else:
        print(count)
        print("no stable")


stable_extension = walkaaf(G, max_flips= 4000 ,max_tries=1000);
# stable_extension = walkaaf_probability_g(G, 0.8, max_flips=2*G.number_of_nodes())

# Then
# Two cases because there exists one stable extension but the algorithm can just find one
if stable_extension:
    print(stable_extension)
else:
    print("no stable")



