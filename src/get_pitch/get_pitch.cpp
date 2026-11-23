/// @file

#include <iostream>
#include <fstream>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <algorithm>
#include <array>

#include "wavfile_mono.h"
#include "pitch_analyzer.h"

#include "docopt.h"

#define FRAME_LEN   0.030 /* 30 ms. */
#define FRAME_SHIFT 0.015 /* 15 ms. */

using namespace std;
using namespace upc;

static const char USAGE[] = R"(
get_pitch - Pitch Estimator 

Usage:
    get_pitch [options] <input-wav> <output-txt>
    get_pitch (-h | --help)
    get_pitch --version

Options:
    -v, --voiced FLOAT    Llindar de probabilitat de sonoritat del classificador logístic
                          (0 = fer servir la regla de llindars -m, -1, -p) [default: 0.6]
    -m, --rmaxnorm FLOAT  Llindar de r[lag]/r[0] per a decisió sord/sonor [default: 0.4]
    -1, --r1norm FLOAT    Llindar de r[1]/r[0] per a decisió sord/sonor [default: 0.0]
    -p, --pot FLOAT       Llindar de potència (dB) relatiu a la trama més forta [default: -100]
    -c, --clip FLOAT      Nivell de center clipping (fracció del màxim de la trama) [default: 0.3]
    -l, --lowpass FLOAT   Freqüència de tall del filtre passa-baixos en Hz (0 = sense filtre) [default: 600]
    -w, --median INT      Longitud del filtre de mediana (1 = sense filtre) [default: 3]
    -t, --medtol FLOAT    El filtre de mediana només canvia el valor si s'allunya de la mediana
                          més d'aquesta fracció (0 = sempre) [default: 0.3]
    -f, --framelen FLOAT  Longitud de la trama en ms [default: 30]
    -o, --offset FLOAT    Desplaçament del centre de la trama en ms [default: 0]
    -r, --refine FLOAT    Longitud (ms) dels segments per refinar el període amb correlació
                          normalitzada (0 = sense refinament) [default: 10]
    -F, --features FILE   Escriu les característiques de cada trama en aquest fitxer
    -H, --hamming         Finestra de Hamming (per defecte, rectangular)
    -h, --help  Show this screen
    --version   Show the version of the project

Arguments:
    input-wav   Wave file with the audio signal
    output-txt  Output file: ASCII file with the result of the estimation:
                    - One line per frame with the estimated f0
                    - If considered unvoiced, f0 must be set to f0 = 0
)";

/// Low-pass FIR filter (windowed sinc, Hamming window), zero phase delay
static void lowpass(vector<float> &x, float fc, unsigned int rate) {
  int M = (int) (0.004 * rate) | 1;  // ~4 ms, odd length
  int h = M / 2;
  vector<float> b(M);
  float wc = 2 * M_PI * fc / rate, sum = 0;
  for (int n = 0; n < M; ++n) {
    int k = n - h;
    float s = (k == 0) ? wc / M_PI : sin(wc * k) / (M_PI * k);
    b[n] = s * (0.54F - 0.46F * cos(2 * M_PI * n / (M - 1)));
    sum += b[n];
  }
  for (float &v : b) v /= sum;
  vector<float> y(x.size(), 0.0F);
  for (int i = 0; i < (int) x.size(); ++i) {
    float acc = 0;
    for (int n = 0; n < M; ++n) {
      int j = i + h - n;
      if (j >= 0 && j < (int) x.size()) acc += b[n] * x[j];
    }
    y[i] = acc;
  }
  x.swap(y);
}

int main(int argc, const char *argv[]) {
  /// \DONE Nous paràmetres a la línia d'ordres
  std::map<std::string, docopt::value> args = docopt::docopt(USAGE,
      {argv + 1, argv + argc},  // array of arguments, without the program name
      true,    // show help if requested
      "2.0");  // version string

  std::string input_wav = args["<input-wav>"].asString();
  std::string output_txt = args["<output-txt>"].asString();
  float rmaxnorm_th = stof(args["--rmaxnorm"].asString());
  float r1norm_th = stof(args["--r1norm"].asString());
  float pot_th = stof(args["--pot"].asString());
  float voiced_th = stof(args["--voiced"].asString());
  float clip = stof(args["--clip"].asString());
  float fc = stof(args["--lowpass"].asString());
  int median_len = stoi(args["--median"].asString());
  float med_tol = stof(args["--medtol"].asString());
  float refine_ms = stof(args["--refine"].asString());
  float frame_ms = stof(args["--framelen"].asString());
  float offset_ms = stof(args["--offset"].asString());
  bool hamming = args["--hamming"].asBool();

  // Read input sound file
  unsigned int rate;
  vector<float> x;
  if (readwav_mono(input_wav, rate, x) != 0) {
    cerr << "Error reading input file " << input_wav << " (" << strerror(errno) << ")\n";
    return -2;
  }

  int n_len = rate * frame_ms / 1000;
  int n_shift = rate * FRAME_SHIFT;
  // Number of frames: the same as with the original 30 ms frames, so the output
  // has the same number of lines as the reference files.
  int n_len0 = rate * FRAME_LEN;
  int n_frames = 0;
  for (int i = 0; i + n_len0 < (int) x.size(); i += n_shift) ++n_frames;
  // Frame k is centred at k*15 ms + 15 ms (+ offset), whatever its length is.
  int pad = n_len;  // zeros at both ends, so all frames are inside the signal
  int n_start = pad + (int) (rate * (FRAME_LEN / 2 + offset_ms / 1000)) - n_len / 2;

  // Define analyzer
  PitchAnalyzer analyzer(n_len, rate, hamming ? PitchAnalyzer::HAMMING : PitchAnalyzer::RECT, 50, 500,
                         rmaxnorm_th, r1norm_th, pot_th, clip);

  // Original signal (without low-pass filter), used for some voicing features
  vector<float> xo(x);
  xo.insert(xo.begin(), pad, 0.0F);
  xo.insert(xo.end(), 2 * pad, 0.0F);

  /// \DONE Preprocessat: filtre passa-baixos
  if (fc > 0)
    lowpass(x, fc, rate);
  x.insert(x.begin(), pad, 0.0F);
  x.insert(x.end(), 2 * pad, 0.0F);

  // Reference power: the strongest frame of the file (so the threshold does
  // not depend on the recording level)
  float pot_max = -1e10;
  vector<float>::iterator iX;
  for (int k = 0; k < n_frames; ++k) {
    iX = x.begin() + n_start + k * n_shift;
    pot_max = max(pot_max, analyzer.frame_power(iX, iX + n_len));
  }
  analyzer.set_pot_ref(pot_max);

  // Iterate for each frame: features and pitch candidate
  vector<FrameFeatures> feats;
  vector<array<float, 3>> raw;  // full band: relative power (dB), r[1]/r[0], zero crossing rate
  for (int k = 0; k < n_frames; ++k) {
    iX = x.begin() + n_start + k * n_shift;
    feats.push_back(analyzer.analyze(iX, iX + n_len));
    feats.back().pot -= pot_max;  // power relative to the strongest frame
    // Features of the original signal (full band)
    vector<float>::const_iterator iO = xo.begin() + n_start + k * n_shift;
    double e = 0, r1 = 0; unsigned int nzc = 0;
    for (int n = 0; n < n_len; ++n) {
      e += iO[n] * iO[n];
      if (n > 0) { r1 += iO[n] * iO[n-1]; if ((iO[n] >= 0) != (iO[n-1] >= 0)) ++nzc; }
    }
    raw.push_back({(float) (10 * log10(e + 1e-10)), (float) (r1 / (e + 1e-10)), (float) nzc / (n_len - 1)});
  }
  float rawpot_max = -1e10;
  for (auto &v : raw) rawpot_max = max(rawpot_max, v[0]);
  for (auto &v : raw) v[0] -= rawpot_max;

  if (args["--features"]) {
    ofstream fs(args["--features"].asString());
    for (unsigned int k = 0; k < feats.size(); ++k) {
      const FrameFeatures &ff = feats[k];
      fs << ff.pot << ' ' << ff.r1norm << ' ' << ff.rmaxnorm << ' ' << ff.nccf << ' ' << ff.zcr << ' ' << ff.f0
         << ' ' << raw[k][0] << ' ' << raw[k][1] << ' ' << raw[k][2] << '\n';
    }
  }

  /// \DONE Decisió sord/sonor.
  /// Classificador de regressió logística entrenat amb la base de dades pitch_db/train.
  /// Per a cada trama fa servir 8 característiques: potència relativa, r[1]/r[0], r[lag]/r[0],
  /// nccf i zcr del senyal filtrat, i potència relativa, r[1]/r[0] i zcr del senyal original.
  /// Es fan servir les de la trama actual i les de les dues anteriors i dues posteriors
  /// (5 x 8 = 40 pesos + terme independent).
  static const int NF = 8;
  static const int OFFS[5] = {0, 2, 1, -1, -2};  // trames: k, k+2, k+1, k-1, k-2
  static const float W[5][NF] = {
    {-0.0361933F, 38.8041F, 2.32819F, 1.85091F, 13.0927F, 0.0450061F, -0.864831F, -2.80501F},
    {0.0115949F, 18.0919F, -1.63622F, 1.3584F, -26.4076F, 0.0158867F, -0.378127F, -1.94958F},
    {0.0625253F, -6.56669F, -0.625628F, 1.87166F, -6.68702F, 0.0238843F, -1.71538F, -3.38568F},
    {0.218176F, -3.08433F, 3.07662F, 1.58884F, -22.2711F, -0.0682284F, 0.331512F, -0.100415F},
    {-0.0187213F, -1.74845F, -1.65701F, 1.00823F, 36.7328F, -0.00147695F, 0.243429F, 0.443494F}
  };
  static const float B = -43.2825F;
  auto fvec = [&](int k) {
    const FrameFeatures &ff = feats[k];
    return array<float, NF>{ff.pot, ff.r1norm, ff.rmaxnorm, ff.nccf, ff.zcr, raw[k][0], raw[k][1], raw[k][2]};
  };

  vector<float> f0;
  for (int k = 0; k < n_frames; ++k) {
    const FrameFeatures &ff = feats[k];
    bool uv;
    if (voiced_th > 0) {
      float z = B;
      for (int c = 0; c < 5; ++c) {
        int j = k + OFFS[c];
        if (j < 0 || j >= n_frames) j = k;  // at the ends, the current frame is used
        array<float, NF> v = fvec(j);
        for (int i = 0; i < NF; ++i)
          z += W[c][i] * v[i];
      }
      float prob = 1.0F / (1.0F + exp(-z));
      uv = prob < voiced_th;
    } else {
      uv = ff.pot < pot_th || ff.r1norm < r1norm_th || ff.rmaxnorm < rmaxnorm_th;
    }
    f0.push_back(uv ? 0.0F : ff.f0);
  }

  /// \DONE Refinament fi del període.
  /// La trama de 30 ms és necessària per trobar el període amb fiabilitat, però si el
  /// pitch varia dins de la trama el resultat és una mitjana. Un cop tenim el candidat T,
  /// busquem el desplaçament l (entre T-3 i T+3 mostres) que maximitza la correlació
  /// normalitzada entre dos segments curts (refine_ms), un a cada costat del centre de
  /// la trama i separats l mostres, i interpolem amb una paràbola.
  if (refine_ms > 0) {
    int N = rate * refine_ms / 1000;
    for (int k = 0; k < n_frames; ++k) {
      if (f0[k] == 0) continue;
      int c = n_start + k * n_shift + n_len / 2;  // centre of the frame
      int L = (int) round(rate / f0[k]);
      auto nc = [&](int l) {
        int a0 = c - N / 2 - l / 2;
        double ab = 0, aa = 0, bb = 0;
        for (int n = 0; n < N; ++n) {
          float a = x[a0 + n], b = x[a0 + l + n];
          ab += a * b; aa += a * a; bb += b * b;
        }
        return (float) (ab / sqrt(aa * bb + 1e-20));
      };
      int lmin = max(2, L - 3), lmax = L + 3;
      vector<float> v;
      for (int l = lmin; l <= lmax; ++l) v.push_back(nc(l));
      int i = max_element(v.begin(), v.end()) - v.begin();
      float T = lmin + i;
      if (i > 0 && i + 1 < (int) v.size()) {
        float den = v[i-1] - 2 * v[i] + v[i+1];
        if (den < 0) T += 0.5F * (v[i-1] - v[i+1]) / den;
      }
      f0[k] = rate / T;
    }
  }

  /// \DONE Postprocessat: filtre de mediana.
  /// - No recursiu: la mediana es calcula sempre sobre els valors ORIGINALS (f0) i el
  ///   resultat s'escriu en un vector nou (f0f). Si escrivíssim a f0 mateix, els valors
  ///   ja filtrats entrarien a la finestra de les trames següents (filtre recursiu).
  /// - Només es fa servir quan l'error és gran: si el valor s'allunya de la mediana menys
  ///   de med_tol (en relatiu), es deixa tal qual, perquè quan no hi ha error la mediana
  ///   empitjora l'estimació fina.
  /// - Només es modifiquen trames sonores i la mediana es calcula amb les veïnes sonores.
  if (median_len > 1) {
    int h = median_len / 2;
    vector<float> f0f(f0);
    for (int i = 0; i < (int) f0.size(); ++i) {
      if (f0[i] == 0) continue;
      vector<float> v;
      for (int k = i - h; k <= i + h; ++k)
        if (k >= 0 && k < (int) f0.size() && f0[k] > 0) v.push_back(f0[k]);
      nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      float med = v[v.size() / 2];
      if (fabs(f0[i] - med) > med_tol * med)
        f0f[i] = med;
    }
    f0.swap(f0f);
  }

  // Write f0 contour into the output file
  ofstream os(output_txt);
  if (!os.good()) {
    cerr << "Error reading output file " << output_txt << " (" << strerror(errno) << ")\n";
    return -3;
  }

  os << 0 << '\n'; // pitch at t=0
  for (iX = f0.begin(); iX != f0.end(); ++iX)
    os << *iX << '\n';
  os << 0 << '\n'; // pitch at t=Dur

  return 0;
}
