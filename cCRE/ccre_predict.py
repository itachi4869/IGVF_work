import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

def load_data(inFile):
    
    sequences = []
    with open(inFile,'r') as f:
        cur_seq = ''
        for line in f:
            if '>' in line:
                sequences.append(cur_seq)
                cur_seq = ''
            else:
                cur_seq += line.strip()
    sequences.append(cur_seq)
    sequences.pop(0) #the first element in the list is just empty

    print(f'#sequences of prediction: {len(sequences)}')
    print(f'length of the first sequence: {len(sequences[0])}')
    
    # Remove sequences with N or n
    # 5 of them are not 300bp long, also removed.
    sequences = [seq for seq in sequences if 'N' not in seq and 'n' not in seq and (len(seq) == 300)]
    print(f'Number of sequences without N or n: {len(sequences)}')

    #sequences = [seq for seq in sequences if set(seq) <= {'A', 'T', 'C', 'G'}]
    #print(f'Number of sequences with only A, T, C, G: {len(sequences)}')

    return sequences

def oneHot(seqs):

    ALPHA = {'A':0, 'C':1, 'G':2, 'T':3}
    numSeqs = len(seqs)
    L = len(seqs[0])
    X = np.zeros((numSeqs, L, 4))
    for j, seq in enumerate(seqs):
        for i in range(L):
            c = seq[i].upper() # Convert a,t,c,g to A,T,C,G
            cat = ALPHA.get(c, -1)
            if(cat >= 0): X[j, i, cat] = 1
    return X

def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]

def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")
    if not rng:
        rng = np.random.RandomState()
  

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of thevoriginal characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]
        counters = [0] * len(chars)     

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]

        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]

def main(modelFilestem, faPath, faFileStem, shuffle=False):

    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Enabled memory growth.")
            
            # Uncomment to restrict TensorFlow to specific GPUs
            # tf.config.set_visible_devices(gpus[0], 'GPU')
            
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU found. Running on CPU.")

    # Load data
    faFile = faPath + '/' + faFileStem + '.fasta'
    sequences = load_data(faFile)
    X = oneHot(sequences)
    del sequences

    # Load model
    model = None
    with open(modelFilestem + '.json', "r") as json_file:
        model_json = json_file.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(modelFilestem + '.h5')
    
    if shuffle:
        # Shuffle the sequences
        rseed = 562
        rng = np.random.RandomState(rseed)
        X_shuf = np.concatenate([dinuc_shuffle(seq, num_shufs=1, rng=rng) for seq in X])
        del X
        
        # Predict shuffled sequences
        batchSize = 1024
        pred_shuf = model.predict(X_shuf, batch_size=batchSize, verbose=0).reshape(-1)
        np.savez_compressed(faFileStem + '_shuf.npz', arr=pred_shuf)
        print(f'predictions saved to {faFileStem}_shuf.npz')
    else:
        # Predict raw sequences
        batchSize = 1024
    
        # The model will automatically use GPU if available with this configuration
        pred = model.predict(X, batch_size=batchSize, verbose=0).reshape(-1)
        np.savez_compressed(faFileStem + '.npz', arr=pred)
        print(f'predictions saved to {faFileStem}.npz')

def compute_stats(arr, bins=100):
    m = np.mean(arr)
    v = np.var(arr)
    counts, bin_edges = np.histogram(arr, bins=bins)
    mode = (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts)+1]) / 2
    return m, mode, v

def plot_density(pred, pred_shuf, pred_non):

    # Compute statistics for each predictions array
    m_pred, mode_pred, v_pred = compute_stats(pred)
    m_shuf, mode_shuf, v_shuf = compute_stats(pred_shuf)
    m_non, mode_non, v_non = compute_stats(pred_non)

    # Kolmogorov-Smirnov test
    rs_statistic, rs_pvalue = stats.ks_2samp(pred, pred_shuf)
    n_statistic, n_pvalue = stats.ks_2samp(pred, pred_non)
    rsn_statistic, rsn_pvalue = stats.ks_2samp(pred_shuf, pred_non)
    print(rs_statistic, rs_pvalue)
    print(n_statistic, n_pvalue)
    print(rsn_statistic, rsn_pvalue)

    # Test whether the activity of non-cCREs is significantly lower than that of randomly shuffled cCREs
    # Use one-sided mann-whitney U test
    u_statistic, u_pvalue = stats.mannwhitneyu(pred_non, pred_shuf, alternative='less')
    print(u_statistic, u_pvalue)

    # Build updated labels including statistics
    label_pred = f'cCREs (mean={m_pred:.2f}, mode={mode_pred:.2f}, var={v_pred:.2f})'
    label_shuf = f'Randomly Shuffled cCREs (mean={m_shuf:.2f}, mode={mode_shuf:.2f}, var={v_shuf:.2f})'
    label_non = f'GC Matched Non-cCREs (mean={m_non:.2f}, mode={mode_non:.2f}, var={v_non:.2f})'

    # Prepare the figure and axis
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Plot density curves on the axis
    sns.kdeplot(pred, label=label_pred, fill=False, ax=ax)
    sns.kdeplot(pred_shuf, label=label_shuf, fill=False, ax=ax)
    sns.kdeplot(pred_non, label=label_non, fill=False, ax=ax)
    ax.set_xlabel('Predicted Activity')
    ax.set_ylabel('Density')
    ax.set_title('Density Plot of Predicted Activity')
    ax.set_xlim(-2, 2)
    
    # Place legend at the upper right of the axes
    ax.legend(loc='upper right', fontsize=9)
    
    # Prepare KS p-value text
    rs_pvalue_text = '< 1e-4' if rs_pvalue < 1e-4 else f'{rs_pvalue:.3e}'
    n_pvalue_text = '< 1e-4' if n_pvalue < 1e-4 else f'{n_pvalue:.3e}'
    rsn_pvalue_text = '< 1e-4' if rsn_pvalue < 1e-4 else f'{rsn_pvalue:.3e}'
    ks_text = (f'KS test (cCREs vs shuffled cCREs): {rs_pvalue_text}\n'
               f'KS test (cCREs vs non-cCREs): {n_pvalue_text}\n'
               f'KS test (shuffled cCREs vs non-cCREs): {rsn_pvalue_text}')
    
    # Place KS test annotation at the upper left of the axes
    plt.text(0.05, 0.95, ks_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('K562-7_cCRE_origin_rshuf_gcnon_density.png', bbox_inches='tight')

def plot_scatter(pred, pred_shuf):
    """
    Plot a scatter plot comparing predictions from original and shuffled sequences.
    The plot will have equal axis limits with a y=x reference line.
    """
    plt.figure(figsize=(8, 8))  # square figure
    plt.scatter(pred, pred_shuf, alpha=0.5, edgecolors='none')
    plt.xlabel('cCREs')
    plt.ylabel('Random Shuffled cCREs')
    plt.title('Scatter Plot of Predicted Activity')

    # Determine common axis limits based on the data range
    min_val = min(np.min(pred), np.min(pred_shuf))
    max_val = max(np.max(pred), np.max(pred_shuf))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # Add the y=x line as a red dashed line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
    plt.legend()

    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.savefig('K562-7_cCRE_origin_rshuf_scatter.png')

if __name__ == '__main__':

    '''
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python plot_dist.py <modelFilestem> <inFile> <faFile>")
        sys.exit(1)

    (modelFileStem, faPath, faFileStem) = sys.argv[1:]
    main(modelFileStem, faPath, faFileStem, shuffle=True)
    '''
    
    # Load predictions
    pred = np.load('K562_new_res/all_cCREs.npz')['arr']
    pred_shuf = np.load('K562_new_res/all_cCREs_shuf.npz')['arr']
    pred_non = np.load('K562_new_res/primary_GC_matched_non_cCREs.npz')['arr']

    # Count the number of predictions
    num_pred = len(pred)
    num_pred_shuf = len(pred_shuf)
    num_pred_non = len(pred_non)
    print(f'Number of predictions: {num_pred}')
    print(f'Number of shuffled predictions: {num_pred_shuf}')
    print(f'Number of GC-matched non-cCREs: {num_pred_non}')

    # Change predictions from natural log base to log2 base
    pred = pred / np.log(2)
    pred_shuf = pred_shuf / np.log(2)
    pred_non = pred_non / np.log(2)

    #plot_scatter(pred, pred_shuf)
    plot_density(pred, pred_shuf, pred_non)
