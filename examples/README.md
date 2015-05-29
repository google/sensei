This directory contanins sample configuration files to use with Sensei.

* basic.config.Flag contains a very simple configuration which reads data,
  performs 100 iterations of training and outputs the trained model.
* scoring.config.Flag reads a previously trained model and scores the data.
  Note that for scoring to work you need to provide the name of a feature which
  will be used as identification number for a row.
* prefixes.config.Flag trains the model on a feature set which is a Cartesian
  product of all the features beginning with letter "a" and all the features
  beginning with letter "b". Please note that, contrary to what is usually done,
  we are not adding one-dimensional features or the bias feature. This is to
  demonstrate how the feature set specification works.
* pruning.config.Flag trains a model, removes the worst scoring features from
  the feature set and performs additional training
* exploration.config.Flag trains a model, adds promising new features to the
  feature set and performs additional training
* regularization.config.Flag showcases the regularization options
* debugging.config.Flag turns on logging and dumps data and model into the logs


