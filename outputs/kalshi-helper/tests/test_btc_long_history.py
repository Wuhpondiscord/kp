import io
import unittest
import zipfile
from helper.btc_long_history import build_rows,parse_archive,timestamp
from helper.btc_research import unix
from helper.core import stamp


class LongHistoryTests(unittest.TestCase):
    def archive(self,lines):
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:z.writestr('prices.csv','\n'.join(lines))
        return out.getvalue()

    def line(self,t,unit=1000000):
        return f'{t*unit},100,102,99,101,10,{(t+60)*unit-1},0,1,0,0,0'

    def test_timestamp_units_and_ohlcv_conversion(self):
        t=unix('2025-01-01')
        for unit in [1000,1000000]:
            self.assertEqual(timestamp(t*unit),t)
            bars=parse_archive(self.archive([self.line(t,unit)]),'2025-01')
            self.assertEqual(bars[t],[t,99,102,100,101,10])
        with self.assertRaises(ValueError):timestamp(t)

    def test_wrong_month_duplicate_or_interval_fails(self):
        t=unix('2025-01-01');line=self.line(t)
        for raw,month in [(self.archive([line]),'2025-02'),(self.archive([line,line]),'2025-01'),
                          (self.archive([line.replace(str((t+60)*1000000-1),str((t+120)*1000000-1))]),'2025-01')]:
            with self.assertRaises(ValueError):parse_archive(raw,month)

    def test_future_label_changes_never_enter_features(self):
        t=unix('2025-01-01')
        bars={s:[s,99,101,100,100,10] for s in range(t,t+4*3600,60)}
        before,_=build_rows(bars);r=before[0]
        close=int(stamp(r['close_time']).timestamp())
        self.assertEqual(r['y'],1)  # equality is YES for the declared proxy
        bars[close-60]=[close-60,98,101,100,99,10]
        after,_=build_rows(bars)
        self.assertEqual(after[0]['y'],0)
        self.assertEqual(r['x'],after[0]['x'])
        self.assertIn('not Kalshi',r['label_source'])
