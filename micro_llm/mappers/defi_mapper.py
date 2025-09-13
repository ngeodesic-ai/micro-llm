from micro_llm.mappers.defi_rules import rule_prior, PRIMS

def map_intent(text, model, threshold=0.7, rule_boost=1.0):
    # your existing inference
    pred_label, conf, prob = model_predict(text, model)

    # rules prior (sparse; 0s if nothing matched)
    r_prior = rule_prior(text, boost=rule_boost)

    if conf >= threshold and pred_label in PRIMS:
        # trust model
        prior = {k: 0.0 for k in PRIMS}; prior[pred_label] = 1.0
        top1 = pred_label
    elif sum(r_prior.values()) > 0:
        # model unsure → lean on rules (safe because they’re narrow)
        prior = r_prior
        top1  = max(prior, key=prior.get)
    else:
        # abstain-friendly default
        prior = {k: 0.0 for k in PRIMS}
        top1  = None

    # publish to aux so the benchmark sees it
    return top1, conf, prior
