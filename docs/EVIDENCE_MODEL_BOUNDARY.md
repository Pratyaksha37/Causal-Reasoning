# Evidence-to-model boundary

The project accepts `BMICat → Outcome` as a literature-derived **structural prior**, with two supporting sentences from two PubMed records. It is a valid directed acyclic graph, not independently proven biological causation. Human validation is pending.

The integrated runner rebuilds this evidence from the saved predictions and passes the resulting GraphML and citation JSON directly into Bayesian construction. Observational association claims stay in a separate undirected graph; they are never converted to causal arrows to enlarge the network.

The eight canonical modelling variables are AgeCat, Sex, Education, Income, Smoker, PhysActivity, BMICat and Outcome. The default model includes only the supported two-node prior. Missing causal structure cannot be repaired merely by fitting CPDs or getting good predictive accuracy.

BRFSS parameters describe unweighted respondents, with Outcome combining prediabetes and diabetes. Literature T2DM/abdominal-obesity claims and survey BMI/combined-outcome measures are imperfectly aligned. These measurement and transport assumptions remain disclosed.

Exploratory models require explicit configuration and edge rationales. Intervention output stays a hypothetical model calculation unless causal identification assumptions and empirical support are explicitly supplied; even then real-world validity is not established by software.

The registry's historical `is_phase2_target` keys and graph-export compatibility function names remain readable to preserve old clients. They do not imply a separate package or project.
