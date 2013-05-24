# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import warnings
warnings.simplefilter('ignore')
import numpy, scipy.sparse, librosa, CDL
import numpy.fft
import scipy.signal
import librosa_analyzer

# <codecell>

def get_spectrogram_samples(filename, mels=128, sample_width=8, pad=False):
    # Load the audio data
    (y, sr) = librosa.load(filename, sr=22050)
    
    # Build mel spectrogram
    # hop=256 => 12ms @ 22050Hz
    S       = librosa.feature.melspectrogram(y, sr, n_mels=mels, n_fft=2048, hop_length=256, fmax=8000)
    
    # Normalize
    S       = S - S.min()
    S       = S / S.max()
    
    N       = S.shape[1] / sample_width
    
    # Truncate any dangling partial frames
    S       = S[:, :(sample_width * N)]
    
    # Reshape into frames
    S       = S.reshape( (S.shape[0], sample_width, N), order='F')
    
    return S.swapaxes(1,2).swapaxes(0,1)

# <codecell>

def reconstruct(D, A):
    
    X = 0.0
    
    def myconv(s1, s2):
        z1 = numpy.fft.fft2(s1)
        z2 = numpy.fft.fft2(s2)
        return numpy.fft.ifft2(z1 * z2).real
    
    
    for i in range(D.shape[0]):
        X = X + myconv(D[i], A[i])
        
    return X
    
def show_activation(Aspace):
    m = Aspace.shape[0]
    w = numpy.floor(m**0.5)
    
    vmax = Aspace.max()
    vmin = 0
    figure()
    for i in range(m):
        subplot(w, numpy.ceil(float(m) / w), i+1)
        imshow(Aspace[i], aspect='auto', origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
        title('Activation %2d' % (i+1))
        axis('off')
        pass
    pass

def show_codebook(D, pad=False):
    
    M, H, W = D.shape
    
    D = D.swapaxes(0,1).swapaxes(1,2)
    D = D.reshape( (D.shape[0], -1), order='F')
    
    imshow(D, aspect='auto', origin='lower', interpolation='none')
    
    nmels    = D.shape[0]
    binfreqs = librosa.feature.mel_frequencies(n_mels=nmels)
    yticks(range(0, nmels+1, nmels/4), map(lambda x: '%dHz' % int(x), binfreqs[1:-1:(nmels/4)]))
    axis('tight')
    colorbar()
    
    vlines(numpy.arange(-0.5, M * W - 0.5, W), 0, H, colors='w', linestyles='dotted')
    
    title('Dictionary - D [%d codewords, %d frames each]' % (M, W))
    xticks(range(W/2, M * W, W), range(1, M+1))
    xlabel('Codeword #')
    pass

# <codecell>

def activation_vertical_marginals(A):
    return A.sum(axis=2).reshape( (A.shape[0], -1))

def activation_horizontal_marginals(A):
    return A.sum(axis=3).reshape( (A.shape[0], -1))

def activation_marginals(A):
    return np.concatenate( (A.sum(axis=2), A.sum(axis=3)), axis=2).reshape( (A.shape[0], -1))

def activation_maxpool(A):
    return np.concatenate( (A.max(axis=2), A.max(axis=3)), axis=2).reshape( (A.shape[0], -1))
import scipy.spatial

def selfsim(X):
    #return 1 - scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric='cosine') )
    return 1-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric='correlation'))

# <codecell>

(H, W) = (64, 8)
PAD    = False

X = get_spectrogram_samples('/home/bmcfee/data/CAL500/wav/art_tatum-willow_weep_for_me.wav', mels=H, sample_width=W, pad=PAD)
X = X.astype(numpy.float32)
N = X.shape[0]

ntrain = (3 * N)/10
Xtrain = X[:ntrain,:,:]
Xtest  = X[ntrain:,:,:]

# <codecell>

# Set up the encoder
reload(CDL)
reload(CDL._cdl)
warnings.simplefilter('ignore')
coder = CDL.ConvolutionalDictionaryLearning(n_atoms=4, alpha=1e-4, n_iter=1, verbose=True, chunk_size=16, pad_data=PAD, n_jobs=4)
print coder

# <codecell>

coder.fit(Xtrain)

# <codecell>

show_codebook(coder.components_, pad=PAD)

# <codecell>

#Xtest = X
Xtest = X[ntrain:]
#Xtest = Xtest[:100]
Atest = coder.transform(Xtest)

# <codecell>

Xpred = np.zeros_like(Xtest)
for i in range(Xtest.shape[0]):
    Xpred[i] = reconstruct(coder.components_, Atest[i])

# <codecell>

figure(figsize=(16,6))
subplot(1,2,1)
show_codebook(librosa.logamplitude(Xpred))
xticks([])
title('Reconstruction')
xlabel('')
subplot(1,2,2)

show_codebook(librosa.logamplitude(Xtest))
xticks([])
title('Original')
xlabel('')

# <codecell>

for i in xrange(min(Atest.shape[0], 4)):
    show_activation(Atest[i])
    pass

# <codecell>

Ztest = activation_horizontal_marginals(Atest)

# <codecell>

figure(figsize=(16,10))
subplot(221)
imshow(Ztest.T, interpolation='none', aspect='auto')
xticks([]), yticks([])
ylabel('pool(A)')

subplot(222)
XV = np.hstack([Xtest[i] for i in range(Ztest.shape[0])])
M = librosa.feature.mfcc(librosa.logamplitude(XV), d=64)
imshow(librosa.logamplitude(XV), aspect='auto', origin='lower', interpolation='none')
xticks([]), yticks([])
ylabel('spectrogram')

subplot(223)

imshow(selfsim(Ztest), interpolation='none', origin='lower', aspect='auto')
xticks([]), yticks([])
ylabel('pool(A) kernel')


subplot(224)
imshow(selfsim(M.T), interpolation='none', origin='lower', aspect='auto')
xticks([]), yticks([])
ylabel('melspec correlation')
pass

