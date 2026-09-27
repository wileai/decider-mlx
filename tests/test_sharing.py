import importlib.util
import mlx.core as mx

def test_shared_prefix_and_right_padding_preserve_slot_hidden():
    from decider_mlx import native
    assert hasattr(native,'shared_hidden'), 'within-request prefix sharing missing'
    from mlx_lm.models.qwen3_5 import Model,ModelArgs
    mx.random.seed(17)
    m=Model(ModelArgs.from_dict(dict(model_type='qwen3_5_text',hidden_size=64,intermediate_size=128,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=32,vocab_size=128,linear_num_key_heads=1,linear_num_value_heads=1,linear_key_head_dim=128,linear_value_head_dim=128,tie_word_embeddings=True)))
    m.eval()
    items=[{'ids':[1,2,3,4,5,6,7,8],'slots':[7],'nopts':[2]},{'ids':[1,2,3,4,9,10],'slots':[5],'nopts':[3]}]
    hs=native.shared_hidden(m,items,0)
    separate=mx.concatenate([m.language_model.model(mx.array([x['ids']]))[:,-1,:] for x in items],axis=0)
    mx.eval(hs,separate)
    assert mx.allclose(hs,separate,atol=2e-5,rtol=2e-5).item()
    padded=m.language_model.model(mx.array([items[0]['ids'],items[1]['ids']+[0,0]]))
    assert mx.allclose(padded[1,5],separate[1],atol=2e-5,rtol=2e-5).item()

def test_pointer_temperature_and_invalid_option_mask():
    from decider_mlx import native
    assert hasattr(native,'pointer_probs'), 'tested pointer readout missing'
    h=mx.array([[1.,2.],[3.,4.]])
    w=mx.array([[.1,.2],[.3,.4],[.5,.6]])
    p=native.pointer_probs(h,w,[2,3],1.3)
    expected=mx.softmax((h@w.T)/1.3,axis=-1)
    assert p[0,2].item()==0
    assert mx.allclose(p[1],expected[1],atol=1e-7).item()
    assert abs(p[0,0].item()+p[0,1].item()-1)<1e-7
