"""T1: 属性关系后处理（推理侧，不重训）。
T1: Attribute relationship post-processing (inference-only, no retrain).

依据：CKTG (TNNLS 2025) 发现 5 属性之间存在有向依赖关系——某些组合极不可能。
Based on: CKTG (TNNLS 2025) found directed dependencies among attributes.
E.g., negative_network & pigment_network rarely co-exist at high confidence.

输入 presence.json，输出修正后的 presence。
Input: presence.json, output: corrected presence.
"""
from .config import ATTRS_JSON, STATUS_HI, STATUS_LO


# 从训练集统计的属性成对条件概率（近似的共现规则）
# Approximate co-occurrence rules from training set statistics
# Key: (attr_a, attr_b) → if attr_a is present, what likely happens to attr_b?
# 'suppress': if a is present, reduce b's prob
# 'boost': if a is present, increase b's prob
ATTR_RELATIONS = {
    # 负性网和色素网极少同时出现 / negative & pigment rarely co-exist
    ('negative_network', 'pigment_network'): ('suppress', 0.3),
    ('pigment_network', 'negative_network'): ('suppress', 0.3),
    # 条纹常在色素网附近 / streaks often near pigment
    ('pigment_network', 'streaks'): ('boost', 0.1),
    # 小球和色素网常共存 / globules & pigment often co-occur
    ('pigment_network', 'globules'): ('boost', 0.05),
}


def apply_attr_rules(presence, rules=None):
    """对 presence 字典应用属性关系规则（推理侧，不改训练）。
    Apply attribute relationship rules to presence dict (inference-only).
    presence: {id: {attr: {prob, status}}}
    Returns corrected presence dict."""
    if rules is None:
        rules = ATTR_RELATIONS
    corrected = {}
    for iid, attrs in presence.items():
        corr = {a: dict(attrs[a]) for a in attrs}
        for (a, b), (action, weight) in rules.items():
            if a in corr and b in corr:
                pa = corr[a]['prob']
                pb = corr[b]['prob']
                if action == 'suppress' and pa > STATUS_HI:
                    # a is present → suppress b
                    pb = pb * (1 - weight)
                elif action == 'boost' and pa > STATUS_HI:
                    # a is present → boost b slightly
                    pb = min(1.0, pb + weight)
                new_status = 'present' if pb >= STATUS_HI else ('absent' if pb <= STATUS_LO else 'uncertain')
                corr[b] = {'prob': round(pb, 4), 'status': new_status}
        corrected[iid] = corr
    return corrected
