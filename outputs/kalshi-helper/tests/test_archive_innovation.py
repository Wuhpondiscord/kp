import copy,math,pickle,unittest
import numpy as np
from helper.archive_model import ArchiveMaximum
from helper.forecast_archive import observation_innovation
from test_forecast_archive import fixture

class ArchiveInnovationTests(unittest.TestCase):
    def row(self):
        return dict(at='2026-04-04T15:00:00Z',series='KXHIGHNY',observed=True,
            observation_at='2026-04-04T14:30:00Z',observation_available_at='2026-04-04T15:00:00Z',observed_temperature=68)
    def test_matches_temperature_at_observation_instead_of_decision(self):
        r=fixture()
        for m in ('gfs_global','ecmwf_ifs025'):
            values=r['body'][0]['hourly']['temperature_2m_'+m];values[14]=70;values[15]=74;values[16]=140
        f=observation_innovation(r,'KXHIGHNY',self.row())
        self.assertEqual(f['forecast_at_observation_f'],72)
        self.assertEqual(f['observation_innovation_f'],-4)
    def test_future_observation_or_forecast_rejected(self):
        r=self.row();r['observation_available_at']='2026-04-04T15:01:00Z'
        with self.assertRaises(ValueError):observation_innovation(fixture(),'KXHIGHNY',r)
        forecast=fixture();forecast['run_initialized_at']='2026-04-04T12:00:00Z'
        with self.assertRaises(ValueError):observation_innovation(forecast,'KXHIGHNY',self.row())
    def test_missing_does_not_fabricate_zero_error(self):
        r=self.row();r['observed']=False
        self.assertIsNone(observation_innovation(fixture(),'KXHIGHNY',r)['observation_innovation_f'])
        f=fixture();f['body'][0]['hourly']['temperature_2m_gfs_global'][14]=None
        self.assertIsNone(observation_innovation(f,'KXHIGHNY',self.row())['observation_innovation_f'])
    def test_wrong_station_and_units_rejected(self):
        with self.assertRaises(ValueError):observation_innovation(fixture(),'KXHIGHDEN',self.row())
        f=fixture();f['body'][0]['hourly_units']['temperature_2m_gfs_global']='K'
        with self.assertRaises(ValueError):observation_innovation(f,'KXHIGHNY',self.row())
    def test_compact_features_exclude_labels_and_settlement_fields(self):
        row=dict(remaining_forecast_max=75,forecast_complete=True,series='KXHIGHNY',model_disagreement_f=2,
            gfs_remaining_max=74,ecmwf_remaining_max=76,observation_innovation_f=-3,observed=False)
        model=ArchiveMaximum(innovation=True)
        _,x=model.inputs([row]);self.assertEqual(x.shape,(1,7))
        _,other=model.inputs([dict(row,y=1,settled_max=1000)])
        np.testing.assert_array_equal(x,other)
        model.theta=np.r_[np.zeros(6),2.,math.log(3)]
        mu,_=model.distribution([row]);self.assertAlmostEqual(mu[0],73.8)
        bins=[dict(row,lower=a,upper=b) for a,b in [(-math.inf,72),(72,78),(78,math.inf)]]
        self.assertAlmostEqual(sum(model.predict(bins)),1)
        np.testing.assert_array_equal(model.predict(bins),pickle.loads(pickle.dumps(model)).predict(bins))
        with self.assertRaises(ValueError):model.inputs([dict(row,observation_innovation_f=None)])
