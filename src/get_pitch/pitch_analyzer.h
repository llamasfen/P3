/// @file

#ifndef PITCH_ANALYZER_H
#define PITCH_ANALYZER_H

#include <vector>
#include <algorithm>
#include <memory>
#include <ffft/FFTReal.h>

namespace upc {
  const float MIN_F0 = 20.0F;    ///< Minimum value of pitch in Hertzs
  const float MAX_F0 = 10000.0F; ///< Maximum value of pitch in Hertzs

  ///
  /// PitchAnalyzer: class that computes the pitch (in Hz) from a signal frame.
  ///
  /// Features of a frame used for the voiced/unvoiced decision
  struct FrameFeatures {
    float pot;      ///< power (dB)
    float r1norm;   ///< r[1]/r[0]
    float rmaxnorm; ///< r[lag]/r[0] (after central clipping)
    float nccf;     ///< normalized cross-correlation at the pitch lag
    float zcr;      ///< zero crossing rate (crossings per sample)
    float f0;       ///< pitch candidate (Hz)
  };

  class PitchAnalyzer {
  public:
    /// Window type
    enum Window {
      RECT,     ///< Rectangular window
      HAMMING   ///< Hamming window
    };

    void set_window(Window type); ///< pre-compute window

  private:
    std::vector<float> window; ///< precomputed window
    unsigned int frameLen,     ///< length of frame (in samples)
      samplingFreq,            ///< sampling rate (in samples per second)
      npitch_min,              ///< minimum value of pitch period, in samples
      npitch_max;              ///< maximum value of pitch period, in samples
    float rmaxnorm_threshold;  ///< threshold for rmaxnorm (voiced if above)
    float r1norm_threshold;    ///< threshold for r1norm (voiced if above)
    float pot_threshold;       ///< threshold for power, in dB relative to pot_ref
    float clip_alpha;          ///< central clipping level (fraction of the frame maximum)
    float pot_ref;             ///< reference power (dB), maximum frame power of the file
    unsigned int fftLen;       ///< FFT length used to compute the autocorrelation (power of 2 >= frameLen + npitch_max)
    std::unique_ptr<ffft::FFTReal<float>> fft; ///< FFT object (precomputed twiddle factors)

    /// Computes correlation from lag=0 to r.size()
    void autocorrelation(const std::vector<float> &x, std::vector<float> &r) const;

    /// Returns the pitch (in Hz) of input frame x
    float compute_pitch(std::vector<float> & x) const;

    /// Returns true is the frame is unvoiced
    bool unvoiced(float pot, float r1norm, float rmaxnorm) const;

  public:
    PitchAnalyzer(unsigned int fLen,                 ///< Frame length in samples
                  unsigned int sFreq,                ///< Sampling rate in Hertzs
                  Window w = PitchAnalyzer::HAMMING, ///< Window type
                  float min_F0 = MIN_F0,             ///< Minimum pitch (Hz)
                  float max_F0 = MAX_F0,             ///< Maximum pitch (Hz)
                  float rmaxnorm_th = 0.4F,          ///< Threshold for rmaxnorm
                  float r1norm_th = 0.0F,            ///< Threshold for r1norm
                  float pot_th = -100.0F,            ///< Threshold for relative power (dB)
                  float clip = 0.0F)                 ///< Central clipping level
    {
      frameLen = fLen;
      samplingFreq = sFreq;
      rmaxnorm_threshold = rmaxnorm_th;
      r1norm_threshold = r1norm_th;
      pot_threshold = pot_th;
      clip_alpha = clip;
      pot_ref = 0.0F;
      set_f0_range(min_F0, max_F0);
      set_window(w);
      // Zero padding to frameLen + npitch_max avoids the circular aliasing of the
      // autocorrelation computed with the DFT
      fftLen = 1;
      while (fftLen < frameLen + npitch_max) fftLen *= 2;
      fft.reset(new ffft::FFTReal<float>(fftLen));
    }

    /// Computes the features and the pitch candidate of a frame (no voicing decision)
    FrameFeatures analyze(std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) const;

    /// Sets the reference power (dB) used by the relative power threshold
    void set_pot_ref(float p) { pot_ref = p; }

    /// Power (dB) of a frame, after windowing
    float frame_power(std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) const;

    /// Operator (): computes the pitch for the given vector x
    float operator()(const std::vector<float> & _x) const {
      if (_x.size() != frameLen)
        return -1.0F;
      std::vector<float> x(_x);
      return compute_pitch(x);
    }

    /// Operator (): computes the pitch for the given "C" vector (float *).
    float operator()(const float * pt, unsigned int N) const {
      if (N != frameLen)
        return -1.0F;
      std::vector<float> x(N);
      std::copy(pt, pt+N, x.begin());
      return compute_pitch(x);
    }

    /// Operator (): computes the pitch for the given vector, expressed by the begin and end iterators
    float operator()(std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) const {
      if (end-begin != frameLen)
        return -1.0F;
      std::vector<float> x(end-begin);
      std::copy(begin, end, x.begin());
      return compute_pitch(x);
    }

    /// Sets pitch range: takes min_F0 and max_F0 in Hz, sets npitch_min and npitch_max in samples
    void set_f0_range(float min_F0, float max_F0);
  };
}
#endif
