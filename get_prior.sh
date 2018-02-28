
if [ ! -d "priors/ade20k/val" ]; then
    mkdir -p priors/ade20k/val
    wget http://vllab1.ucmerced.edu/~whung/SPGC/ade20k_val_prior.tar.gz
    tar -xf ade20k_val_prior.tar.gz -C priors/ade20k/val
    rm -f ade20k_val_prior.tar.gz
fi


if [ ! -d "context_feats/ade20k/val" ]; then
    mkdir -p context_feats/ade20k/val
    wget http://vllab1.ucmerced.edu/~whung/SPGC/ade20k_val_feat.tar.gz
    tar -xf ade20k_val_feat.tar.gz -C context_feats/ade20k/val
    rm -f ade20k_val_feat.tar.gz
fi


