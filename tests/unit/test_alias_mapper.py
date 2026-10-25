"""Unit tests for AliasMapper precision rules, polarity flipping, and unmapped fallbacks."""

from src.mapping.alias_mapper import AliasMapper


def test_phys_activity_mapping_and_polarity_flip() -> None:
    mapper = AliasMapper()

    # Active lifestyle -> PhysActivity=1, polarity_flip=False
    active = mapper.map_mention("physical activity")
    assert active["canonical_node_id"] == "PhysActivity"
    assert active["value_state"] == "1"
    assert active["polarity_flip"] is False

    # Inactive lifestyle -> PhysActivity=0, polarity_flip=True
    inactive = mapper.map_mention("physical inactivity")
    assert inactive["canonical_node_id"] == "PhysActivity"
    assert inactive["value_state"] == "0"
    assert inactive["polarity_flip"] is True

    sedentary = mapper.map_mention("sedentary lifestyle")
    assert sedentary["canonical_node_id"] == "PhysActivity"
    assert sedentary["value_state"] == "0"
    assert sedentary["polarity_flip"] is True


def test_smoker_strict_precision_rules() -> None:
    mapper = AliasMapper()

    # Explicit CDC 100-cigarette lifetime criterion maps to Smoker=1
    smoker_valid = mapper.map_mention("ever smoked at least 100 cigarettes")
    assert smoker_valid["canonical_node_id"] == "Smoker"
    assert smoker_valid["value_state"] == "1"

    never_smoked = mapper.map_mention("never smoked")
    assert never_smoked["canonical_node_id"] == "Smoker"
    assert never_smoked["value_state"] == "0"

    # General smoking mentions must NOT be mapped to Smoker
    assert mapper.map_mention("smoking")["canonical_node_id"] == "unmapped"
    assert mapper.map_mention("cigarette use")["canonical_node_id"] == "unmapped"
    assert mapper.map_mention("tobacco use")["canonical_node_id"] == "unmapped"
    assert mapper.map_mention("smoking history")["canonical_node_id"] == "unmapped"
    assert mapper.map_mention("smoking cessation")["canonical_node_id"] == "unmapped"
    assert mapper.map_mention("current smoker")["canonical_node_id"] == "unmapped"


def test_bmicat_precision_rules() -> None:
    mapper = AliasMapper()

    # Obesity / BMI >= 30 maps to highest BMICat (obese)
    obese_mention = mapper.map_mention("obesity")
    assert obese_mention["canonical_node_id"] == "BMICat"
    assert obese_mention["value_state"] == "obese"

    bmi30_mention = mapper.map_mention("BMI >= 30")
    assert bmi30_mention["canonical_node_id"] == "BMICat"
    assert bmi30_mention["value_state"] == "obese"

    # Overweight maps to overweight
    overweight_mention = mapper.map_mention("overweight")
    assert overweight_mention["canonical_node_id"] == "BMICat"
    assert overweight_mention["value_state"] == "overweight"

    # Normal weight maps to normal
    normal_mention = mapper.map_mention("normal weight")
    assert normal_mention["canonical_node_id"] == "BMICat"
    assert normal_mention["value_state"] == "normal"


def test_outcome_precision_rules() -> None:
    mapper = AliasMapper()

    # Outcome=1 for T2DM / prediabetes or diabetes
    t2dm = mapper.map_mention("type 2 diabetes")
    assert t2dm["canonical_node_id"] == "Outcome"
    assert t2dm["value_state"] == "1"

    prediabetes = mapper.map_mention("prediabetes or diabetes")
    assert prediabetes["canonical_node_id"] == "Outcome"
    assert prediabetes["value_state"] == "1"

    # Outcome=0 for no diabetes / diabetes-free
    no_diabetes = mapper.map_mention("no diabetes")
    assert no_diabetes["canonical_node_id"] == "Outcome"
    assert no_diabetes["value_state"] == "0"

    normoglycemic = mapper.map_mention("normoglycemic")
    assert normoglycemic["canonical_node_id"] == "Outcome"
    assert normoglycemic["value_state"] == "0"


def test_unmapped_fallback_never_force_match() -> None:
    mapper = AliasMapper()
    assert mapper.map_mention("random protein phosphorylation")["canonical_node_id"] == "unmapped"
    assert mapper.map_mention("mitochondrial fission")["canonical_node_id"] == "unmapped"
