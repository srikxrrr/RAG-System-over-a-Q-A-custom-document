# Soil Mechanics: Course Notes (Unit 2 to 4)

*Generic study notes written for this demo. Check formulas against your textbook and lecturer's slides.*

## 1 Index properties

### 1.1 Atterberg limits

The liquid limit LL is the water content at which soil changes from plastic to liquid behaviour. The plastic limit PL is the water content at which it changes from semi-solid to plastic. The plasticity index is PI = LL - PL.

### 1.2 Compaction

The standard Proctor test finds the optimum moisture content (OMC) that gives the maximum dry density. Compacting wet of optimum gives lower strength but more flexibility.

## 2 Effective stress

### 2.1 Terzaghi's principle

The effective stress is the total stress minus the pore water pressure: sigma' = sigma - u. Soil strength and volume change depend on the effective stress, not on the total stress.

### 2.2 Worked idea

At depth z below the ground surface with the water table at the surface, u = gamma_w x z, so sigma' = (gamma_sat - gamma_w) x z, where gamma_w is about 9.81 kN/m3.

## 3 Consolidation

### 3.1 Primary consolidation settlement

For a normally consolidated clay layer of thickness H, the settlement is Sc = Cc H / (1 + e0) x log10((sigma'0 + delta sigma') / sigma'0), where Cc is the compression index and e0 the initial void ratio.

### 3.2 Time rate of consolidation

The time factor is Tv = cv t / Hdr^2, where cv is the coefficient of consolidation and Hdr is the drainage path length. For a double-drained layer Hdr is half the layer thickness.

Table 1 Time factor against average degree of consolidation

| Degree of consolidation U (%) | Time factor Tv |
|---|---|
| 25 | 0.049 |
| 50 | 0.197 |
| 90 | 0.848 |

## 4 Bearing capacity

### 4.1 Terzaghi bearing capacity of a strip footing

The ultimate bearing capacity is q_ult = c Nc + q Nq + 0.5 gamma B Ngamma, where c is cohesion, q the surcharge at foundation level, gamma the unit weight and B the footing width. For a purely cohesive soil with phi = 0 the factor Nc is 5.7.

### 4.2 Factor of safety

The allowable bearing capacity is obtained by dividing the net ultimate bearing capacity by a factor of safety, usually taken as 3 for shallow foundations in these notes.
