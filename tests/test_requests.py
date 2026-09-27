from types import SimpleNamespace
import pytest


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode('utf-8'))


def runtime(upstream):
    from decider_mlx import Decider
    model = object.__new__(Decider)
    model.upstream = upstream
    model.tok = CharacterTokenizer()
    model.max_state_tokens = 2048
    model.cfg = {'temperature': 1.05}
    return model


@pytest.mark.parametrize('override', [{}, {'isolated': True}], ids=['default', 'true'])
def test_score_keeps_all_branches_and_assembles_original_confidence(upstream, override):
    model = runtime(upstream)
    questions = {'q': {'type': 'score', 'instructions': 'How warm?',
                       'criteria': ['cold', 'mild', 'hot'], **override}}
    rqs, index, items = model.items('The temperature is mild.', questions)
    assert len(items) == 3
    assert all(item['nopts'] == [2] for item in items)
    assert all(item['perms'] == [[0, 1]] for item in items)
    answer = upstream.systemone.assemble(rqs, index, [[.9, .1], [.2, .8], [.9, .1]])['q']
    assert answer['score'] == 1.0
    assert answer['confidence'] == .8
    assert answer['level_fit'] == {'0': .1, '1': .8, '2': .1}
    assert answer['fit_mass'] == 1.0


@pytest.mark.parametrize('isolated', [False, None, 0, 1, '', 'private-value', [], {}])
def test_score_rejects_non_true_isolated_before_rendering(upstream, monkeypatch, isolated):
    model = runtime(upstream)
    questions = {'private-id': {'type': 'score', 'instructions': 'private-instructions',
                               'criteria': ['private-low', 'private-mid', 'private-high'],
                               'isolated': isolated}}

    def unexpected_render(*args, **kwargs):
        pytest.fail('Unsupported Score isolation reached upstream rendering')

    monkeypatch.setattr(upstream.systemone, 'render_question', unexpected_render)
    monkeypatch.setattr(upstream.systemone, 'render_state', unexpected_render)
    with pytest.raises(ValueError) as error:
        model.items('private-state', questions)
    assert str(error.value) == 'Score questions require isolated=True'


def test_overlong_state_is_rejected_without_truncation(upstream):
    model = runtime(upstream)
    model.max_state_tokens = 12
    with pytest.raises(ValueError, match='refusing truncation'):
        model.items('x' * 20, {'q': {'type': 'noul', 'instructions': 'Is it warm?'}})


def test_decide_returns_original_assembly_with_request_local_temperature(upstream):
    model = runtime(upstream)
    observed = []
    def score(items, temperature):
        observed.append((len(items), temperature))
        return [[.25, .75]]
    model.score = score
    question = {'q': {'type': 'choice', 'instructions': 'Pick a color.',
                      'criteria': ['red', 'green']}}
    answer = model.decide('The light is green.', question)
    assert answer['answers']['q']['choice'] == 'green'
    assert observed == [(1, 1.05)]
    assert not hasattr(model, 'temperature')
    assert not hasattr(model, 'cache')
