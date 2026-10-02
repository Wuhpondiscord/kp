"""Fixed parent-weight ensembles; no learned mixing or routing."""
import numpy as np

class SeedEnsemble:
    """Three equally weighted seeds of one named model family."""
    def __init__(self,members,name):
        if len(members)!=3:raise ValueError('Require three seeds')
        self.members=list(members);self.name=name
    def predict(self,rows,paths):
        p=np.asarray([m.predict(rows,paths) for m in self.members])
        if p.shape!=(3,len(rows)) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Invalid member probabilities')
        return p.mean(0)

    def predict_event(self,rows,paths):
        from .event_probabilities import validate_event,project_event
        validate_event(rows)
        return project_event(rows,self.predict(rows,paths))

class FusionBlend:
    def __init__(self,original,revised,weather_weight=.5):
        if len(original)!=3 or len(revised)!=3:
            raise ValueError('Require all three seeds from each parent')
        if not np.isfinite(weather_weight) or not 0<=weather_weight<=1:
            raise ValueError('WeatherSignal weight must be finite and between zero and one')
        self.weather_weight=float(weather_weight)
        self.original=list(original);self.revised=list(revised)

    def predict(self,rows,paths):
        predictions=np.asarray([m.predict(rows,paths) for m in self.original+self.revised])
        if predictions.shape!=(6,len(rows)) or not np.isfinite(predictions).all() or np.any((predictions<0)|(predictions>1)):
            raise ValueError('Invalid member probabilities')
        weight=getattr(self,'weather_weight',.5) # Existing saved blends remain 50/50.
        return weight*predictions[:3].mean(axis=0)+(1-weight)*predictions[3:].mean(axis=0)

    def predict_event(self,rows,paths):
        # Normalize each parent first: a 60/40 blend remains exactly 60/40.
        from .event_probabilities import validate_event,project_event
        validate_event(rows)
        weather=SeedEnsemble(self.original,'WeatherSignal').predict_event(rows,paths)
        guard=SeedEnsemble(self.revised,'MarketGuard').predict_event(rows,paths)
        weight=getattr(self,'weather_weight',.5)
        return project_event(rows,weight*weather+(1-weight)*guard)
