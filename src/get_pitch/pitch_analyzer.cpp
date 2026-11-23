/// @file

#include <iostream>
#include <math.h>
#include "pitch_analyzer.h"

using namespace std;

/// Name space of UPC
namespace upc {
  void PitchAnalyzer::autocorrelation(const vector<float> &x, vector<float> &r) const {
    /// \DONE Autocorrelació calculada.
    /// En lloc de la suma directa r[l] = sum x[n] x[n+l], que costa O(N·L), es fa servir el
    /// teorema de Wiener-Khinchin: l'autocorrelació és la transformada inversa de la densitat
    /// espectral de potència |X(k)|^2. Amb la FFT el cost és O(M log M). El senyal s'omple
    /// amb zeros fins a M >= N + L per evitar l'aliasing temporal de la correlació circular.
    /// El resultat és idèntic al de la suma directa (llevat d'errors d'arrodoniment).
    unsigned int M = fftLen;
    vector<float> xp(M, 0.0F), X(M), p(M, 0.0F), rr(M);
    copy(x.begin(), x.end(), xp.begin());
    fft->do_fft(X.data(), xp.data());
    // FFTReal layout: X[0..M/2] real parts, X[M/2+k] imaginary part of bin k (k=1..M/2-1)
    p[0] = X[0] * X[0];
    p[M/2] = X[M/2] * X[M/2];
    for (unsigned int k = 1; k < M/2; ++k)
      p[k] = X[k] * X[k] + X[M/2 + k] * X[M/2 + k];  // |X(k)|^2, imaginary part = 0
    fft->do_ifft(p.data(), rr.data());
    fft->rescale(rr.data());
    for (unsigned int l = 0; l < r.size(); ++l)
      r[l] = rr[l];

    if (r[0] <= 0.0F) // to avoid log() and divide zero
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
    /// \DONE Regla senzilla de decisió sord/sonor amb tres llindars: potència relativa al
    /// màxim del fitxer, r[1]/r[0] i r[lag]/r[0]. És la primera versió de l'estimador; la
    /// versió final (get_pitch.cpp) fa servir un classificador de regressió logística.
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

    /// \DONE Preprocessat: center clipping.
    /// La no linealitat elimina les mostres petites (|x| < clip_alpha·max) i resta el nivell a
    /// la resta. El senyal s'assembla més a un tren de deltes de Dirac (excitació glotal),
    /// s'atenua l'efecte dels formants i el pic de l'autocorrelació al període és més marcat.
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

    /// \DONE Màxim de l'autocorrelació lluny de l'origen.
    /// Es busca el màxim entre npitch_min (període del pitch màxim, 500 Hz) i npitch_max
    /// (període del pitch mínim, 50 Hz). Es fa servir l'autocorrelació esbiaixada, que
    /// atenua els retards llargs i així evita els errors de meitat de pitch.
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
