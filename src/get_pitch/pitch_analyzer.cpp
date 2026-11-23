/// @file

#include <iostream>
#include <math.h>
#include "pitch_analyzer.h"

using namespace std;

/// Name space of UPC
namespace upc {
  void PitchAnalyzer::autocorrelation(const vector<float> &x, vector<float> &r) const {
    for (unsigned int l = 0; l < r.size(); ++l) {
      /// \DONE Autocorrelació calculada
      r[l] = 0.0F;
      for (unsigned int n = 0; n < x.size()-l; ++n)
        r[l] += x[n]*x[n+l];
    }
    if (r[0] == 0.0F) // to avoid log() and divide zero
      r[0] = 1e-10;
  }

  void PitchAnalyzer::set_window(Window win_type) {
    if (frameLen == 0)
      return;

    window.resize(frameLen);

    switch (win_type) {
    case HAMMING:
      /// \DONE Finestra de Hamming
      for (unsigned int n = 0; n < frameLen; ++n)
        window[n] = 0.54F - 0.46F * cos(2 * M_PI * n / (frameLen - 1));
      break;
    case RECT:
    default:
      window.assign(frameLen, 1);
    }
  }

  void PitchAnalyzer::set_f0_range(float min_F0, float max_F0) {
    npitch_min = (unsigned int) samplingFreq/max_F0;
    if (npitch_min < 2)
      npitch_min = 2;

    npitch_max = 1 + (unsigned int) samplingFreq/min_F0;

    // frameLen should include at least 2*T0
    if (npitch_max > frameLen/2)
      npitch_max = frameLen/2;
  }

  float PitchAnalyzer::frame_power(vector<float>::const_iterator begin, vector<float>::const_iterator end) const {
    double p = 0;
    unsigned int i = 0;
    for (auto it = begin; it != end && i < frameLen; ++it, ++i) {
      float v = *it * window[i];
      p += v * v;
    }
    return 10 * log10(p + 1e-10);
  }

  bool PitchAnalyzer::unvoiced(float pot, float r1norm, float rmaxnorm) const {
    /// \DONE Decisió sord/sonor amb tres característiques:
    /// potència relativa al màxim del fitxer, r[1]/r[0] i r[lag]/r[0]
    if (pot - pot_ref < pot_threshold) return true;
    if (r1norm < r1norm_threshold) return true;
    if (rmaxnorm < rmaxnorm_threshold) return true;
    return false;
  }

  FrameFeatures PitchAnalyzer::analyze(vector<float>::const_iterator begin, vector<float>::const_iterator end) const {
    FrameFeatures ff;
    vector<float> x(begin, end);
    unsigned int N = x.size();

    // Window input frame
    for (unsigned int i = 0; i < N; ++i)
      x[i] *= window[i];

    // Features of the original frame: power, r[1]/r[0] and zero crossing rate
    vector<float> r(npitch_max);
    autocorrelation(x, r);
    ff.pot = 10 * log10(r[0]);
    ff.r1norm = r[1] / r[0];
    unsigned int nzc = 0;
    for (unsigned int n = 1; n < N; ++n)
      if ((x[n] >= 0) != (x[n-1] >= 0)) ++nzc;
    ff.zcr = (float) nzc / (N - 1);

    // Central clipping: the signal becomes closer to a train of deltas,
    // so the autocorrelation peak at the pitch period is sharper
    vector<float> xc(x);
    vector<float> r_orig(r);
    if (clip_alpha > 0) {
      float xmax = 0;
      for (float v : xc) xmax = max(xmax, fabs(v));
      float cl = clip_alpha * xmax;
      for (float &v : xc) {
        if (v > cl) v -= cl;
        else if (v < -cl) v += cl;
        else v = 0;
      }
      autocorrelation(xc, r);
    }

    /// \DONE Màxim de l'autocorrelació entre npitch_min i npitch_max
    vector<float>::const_iterator iRMax = max_element(r.begin() + npitch_min, r.begin() + npitch_max);
    unsigned int lag = iRMax - r.begin();
    ff.rmaxnorm = r[lag] / r[0];

    // Normalized cross-correlation between x[0..N-lag) and x[lag..N)
    double c = 0, e0 = 0, el = 0;
    for (unsigned int n = 0; n + lag < N; ++n) {
      c += x[n] * x[n+lag]; e0 += x[n] * x[n]; el += x[n+lag] * x[n+lag];
    }
    ff.nccf = c / sqrt(e0 * el + 1e-20);

    // Refinement of the period. The autocorrelation computed with a finite frame is
    // biased: r[l] is attenuated by (N-l)/N, which moves the maximum towards shorter
    // lags. Around the maximum we use the unbiased values, r[l]/(N-l), and then a
    // parabolic interpolation to obtain a sub-sample estimation of the period.
    // The refinement uses the autocorrelation of the frame without clipping, since
    // clipping distorts the shape of the peak.
    auto ru = [&](unsigned int l) { return r_orig[l] / (float) (N - l); };
    while (lag + 1 < npitch_max && ru(lag + 1) > ru(lag)) ++lag;
    while (lag > npitch_min && ru(lag - 1) > ru(lag)) --lag;
    float flag = lag;
    if (lag > npitch_min && lag + 1 < npitch_max) {
      float a = ru(lag-1), b = ru(lag), cc = ru(lag+1);
      float den = a - 2*b + cc;
      if (den < 0) flag += 0.5F * (a - cc) / den;
    }
    ff.f0 = (float) samplingFreq / flag;
    return ff;
  }

  float PitchAnalyzer::compute_pitch(vector<float> & x) const {
    if (x.size() != frameLen)
      return -1.0F;
    FrameFeatures ff = analyze(x.begin(), x.end());
    if (unvoiced(ff.pot, ff.r1norm, ff.rmaxnorm))
      return 0;
    return ff.f0;
  }
}
