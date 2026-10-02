import pickle
import unittest
import numpy as np
from scipy.special import logit
from helper.residual import ResidualNetwork,date_weights
from helper.training import settings,segment_scores
from helper.evaluation import features


def samples():
    return [dict(p=p,spread=.02,hours_left=12,series='KXHIGHNY',group=str(i//3),y=i%2) for i,p in enumerate([.01,.1,.3,.5,.7,.99])]


class ResidualTests(unittest.TestCase):
    def test_initial_prediction_is_market_and_zero_weight_exact(self):
        rows=samples();model=ResidualNetwork(rows,settings())
        np.testing.assert_allclose(model.predict(rows),[r['p'] for r in rows],atol=1e-12)
        model.fit_epoch(rows);model.weight=0
        np.testing.assert_array_equal(model.predict(rows),[r['p'] for r in rows])

    def test_backprop_matches_numerical_gradient_in_every_layer(self):
        rows=samples();model=ResidualNetwork(rows,settings({'architecture':'16,8'}))
        model.coefs_[-1][:]=.1
        X=model.scaler.transform(features(rows));offset=logit([r['p'] for r in rows]);y=np.array([r['y'] for r in rows]);w=date_weights(rows)
        _,grads=model.loss_grad(X,offset,y,w)
        for param,grad in zip(model.parameters(),grads):
            index=(0,)*param.ndim;old=param[index];eps=1e-6
            param[index]=old+eps;plus=model.loss_grad(X,offset,y,w)[0]
            param[index]=old-eps;minus=model.loss_grad(X,offset,y,w)[0];param[index]=old
            self.assertAlmostEqual(grad[index],(plus-minus)/(2*eps),places=6)

    def test_dates_have_equal_total_weight_despite_quote_count(self):
        rows=samples()+[dict(samples()[0],group='0')]*9;w=date_weights(rows)
        totals=[sum(x for r,x in zip(rows,w) if r['group']==g) for g in ('0','1')]
        self.assertAlmostEqual(*totals)
        self.assertAlmostEqual(float(w.mean()),1)

    def test_learns_known_bias_and_pickle_keeps_predictions(self):
        rows=[dict(samples()[0],p=.2,y=int(i%10<7)) for i in range(200)]
        model=ResidualNetwork(rows,settings({'batch_size':64,'learning_rate':.01}))
        for _ in range(50):model.fit_epoch(rows)
        self.assertGreater(float(model.predict(rows).mean()),.5)
        np.testing.assert_array_equal(model.predict(rows),pickle.loads(pickle.dumps(model)).predict(rows))
