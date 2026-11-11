from saywhat.align import normalize, replacements, tokens


def test_tokens_keep_technical_shapes():
    assert tokens("Run node.js on k8s, then CI/CD.") == ["Run", "node.js", "on", "k8s", "then", "CI/CD"]


def test_normalize_ignores_case_and_punctuation():
    assert normalize("Apply the kubectl manifest.") == normalize("apply THE kubectl manifest")


def test_replacements_extract_changed_spans():
    heard = "Apply the cubectal manifest to the x cluster"
    chosen = "Apply the kubectl manifest to the EKS cluster"
    assert replacements(heard, chosen) == [("cubectal", "kubectl"), ("x", "EKS")]


def test_replacements_multiword_span():
    assert replacements("deploy with cooper netties", "deploy with Kubernetes") == [("cooper netties", "Kubernetes")]
