import pickle, numpy as np
stays = pickle.load(open('./data/ihm/train_ihm-48-notes-missingInd-standardized_stays.pkl','rb'))
s0 = stays[0]
fn = s0.get('feature_names')
print('n feature_names:', None if fn is None else len(fn))
print(fn)
samp = np.concatenate([np.asarray(s['reg_ts']) for s in stays[:1000]], axis=0)
cf = np.isnan(samp).mean(axis=0)
bad = np.where(cf > 0.999)[0]
print('reg_ts cols:', samp.shape[1], 'fully-NaN cols:', bad.tolist())
for i in bad:
    print(' col', i, '->', fn[i] if fn is not None and i < len(fn) else '(no name)')
print('irg_ts cols:', np.asarray(s0['irg_ts']).shape[1])