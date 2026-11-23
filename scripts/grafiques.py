#! /usr/bin/env python3
"""Gràfiques de la memòria de la P3 (estimació de pitch).

Ús: python3 scripts/grafiques.py <directori_sortida> <fitxers_de_dades...>
(vegeu el README per a les ordres concretes que generen les dades)
"""
import sys
import wave
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
FS_SHIFT = 0.015  # desplaçament de trama (s)


def llegeix_wav(nom):
    with wave.open(nom) as f:
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16) / 32768
        return x, f.getframerate()


def autocorrelacio(wav, f0ref, sortida, trama=110):
    """Senyal de 30 ms d'una trama sonora i la seva autocorrelació."""
    x, fs = llegeix_wav(wav)
    ref = np.loadtxt(f0ref)
    N = int(0.030 * fs)
    ini = trama * int(FS_SHIFT * fs) - N // 2
    seg = x[ini:ini + N]
    t = 1000 * np.arange(N) / fs
    r = np.correlate(seg, seg, 'full')[N - 1:] / np.dot(seg, seg)
    lmin, lmax = fs // 500, fs // 50
    lag = lmin + np.argmax(r[lmin:lmax])
    T = 1000 * lag / fs

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5))
    a1.plot(t, seg, lw=1)
    a1.set_title(f'Segment sonor de 30 ms (prueba.wav, t = {trama * FS_SHIFT:.3f} s)')
    a1.set_xlabel('temps (ms)')
    a1.set_ylabel('amplitud')
    imax = np.argmax(seg[:lag])
    for k in range(4):
        tt = t[imax] + k * T
        if tt < t[-1]:
            a1.axvline(tt, color='C3', ls='--', lw=0.8)
    a1.annotate('', xy=(t[imax] + T, 0.9 * seg.max()), xytext=(t[imax], 0.9 * seg.max()),
                arrowprops=dict(arrowstyle='<->', color='C3'))
    a1.text(t[imax] + T / 2, 0.95 * seg.max(), f'T0 = {T:.2f} ms', color='C3', ha='center', va='bottom')

    l = np.arange(len(r))
    a2.plot(1000 * l / fs, r, lw=1)
    a2.axvspan(1000 * lmin / fs, 1000 * min(lmax, N - 1) / fs, color='C2', alpha=0.08,
               label='marge de cerca (50-500 Hz)')
    a2.plot(T, r[lag], 'o', color='C3',
            label=f'màxim secundari: lag = {lag} mostres, f0 = {fs / lag:.1f} Hz')
    a2.set_title('Autocorrelació normalitzada r[l]/r[0]')
    a2.set_xlabel('retard (ms)')
    a2.legend(loc='upper right')
    print(f'autocorrelació: f0 estimat {fs / lag:.1f} Hz, referència {ref[trama]:.1f} Hz')
    fig.tight_layout()
    fig.savefig(sortida, dpi=130)


def caracteristiques(wav, f0ref, ft, sortida):
    """Senyal, potència, r1norm, rmaxnorm, zcr i pitch de referència (com els panells de wavesurfer)."""
    x, fs = llegeix_wav(wav)
    ref = np.loadtxt(f0ref)
    F = np.loadtxt(ft)
    tf = FS_SHIFT * (np.arange(len(F)) + 1)  # centre de cada trama
    tr = FS_SHIFT * np.arange(len(ref))
    sonor = ref > 0

    fig, ax = plt.subplots(6, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(np.arange(len(x)) / fs, x, lw=0.5)
    ax[0].set_ylabel('senyal')
    ax[1].plot(tf, F[:, 0], lw=1)
    ax[1].set_ylabel('potència (dB)\nrel. al màxim')
    ax[2].plot(tf, F[:, 1], lw=1)
    ax[2].set_ylabel('r1norm')
    ax[3].plot(tf, F[:, 2], lw=1)
    ax[3].set_ylabel('rmaxnorm')
    ax[4].plot(tf, F[:, 4] * fs, lw=1)
    ax[4].set_ylabel('ZCR (creu./s)')
    ax[5].plot(tr[sonor], ref[sonor], '.', ms=3)
    ax[5].set_ylabel('f0 ref. (Hz)')
    ax[5].set_xlabel('temps (s)')
    for a in ax:
        a.fill_between(tr, 0, 1, where=sonor, color='C1', alpha=0.12,
                       transform=a.get_xaxis_transform(), step='mid')
    ax[0].set_title('Característiques per a la decisió sonor/sord (ombrejat: trames sonores de la referència)')
    fig.tight_layout()
    fig.savefig(sortida, dpi=130)


def comparacio(f0ref, f0_nostre, f0_rapt, sortida):
    """Comparació del nostre estimador amb RAPT (l'estimador de wavesurfer) i la referència."""
    ref = np.loadtxt(f0ref)
    a = np.loadtxt(f0_nostre)
    b = np.loadtxt(f0_rapt)
    t = FS_SHIFT * np.arange(len(ref))
    nan = lambda v: np.where(v > 0, v, np.nan)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, nan(ref), 'k-', lw=2.5, alpha=0.35, label='referència (laringògraf)')
    ax.plot(t, nan(b), 'C1.', ms=5, label='RAPT / get_f0 (wavesurfer)')
    ax.plot(t, nan(a), 'C0x', ms=4, label='el nostre get_pitch')
    ax.set_xlabel('temps (s)')
    ax.set_ylabel('f0 (Hz)')
    ax.set_title('prueba.wav: estimació de pitch')
    ax.legend(loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(sortida, dpi=130)


def mediana(sortida):
    """Estudi del filtre de mediana: TOTAL en funció de la longitud i del marge."""
    marge = [0, 0.1, 0.2, 0.3, 0.5]
    total = {3: [93.87, 94.25, 94.49, 94.50, 94.50],
             5: [93.47, 94.11, 94.43, 94.48, 94.49],
             7: [93.07, 93.90, 94.40, 94.46, 94.48]}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axhline(94.50, color='k', ls='--', lw=1, label='sense filtre')
    for w, v in total.items():
        ax.plot(marge, v, 'o-', label=f'longitud {w}')
    ax.set_xlabel('marge relatiu a partir del qual actua el filtre (-t)')
    ax.set_ylabel('TOTAL (%)')
    ax.set_title('Filtre de mediana no recursiu (pitch_db/train)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(sortida, dpi=130)


if __name__ == '__main__':
    d = sys.argv[1]
    autocorrelacio('prueba.wav', 'prueba.f0ref', f'{d}/autocorrelacio.png')
    caracteristiques('prueba.wav', 'prueba.f0ref', sys.argv[2], f'{d}/caracteristiques.png')
    comparacio('prueba.f0ref', sys.argv[3], sys.argv[4], f'{d}/comparacio_rapt.png')
    mediana(f'{d}/mediana.png')
