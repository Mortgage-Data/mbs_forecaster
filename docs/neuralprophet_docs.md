# See https://neuralprophet.com/code/time_net.html
# Core Module Documentation
class neuralprophet.forecaster.NeuralProphet(growth: Literal['off', 'linear', 'discontinuous'] = 'linear', changepoints: Optional[list] = None, n_changepoints: int = 10, changepoints_range: float = 0.8, trend_reg: float = 0, trend_reg_threshold: Optional[Union[bool, float]] = False, trend_global_local: str = 'global', trend_local_reg: Optional[Union[bool, float]] = False, yearly_seasonality: Union[Literal['auto'], bool, int] = 'auto', yearly_seasonality_glocal_mode: Union[Literal['auto'], bool, int] = 'auto', weekly_seasonality: Union[Literal['auto'], bool, int] = 'auto', weekly_seasonality_glocal_mode: Union[Literal['auto'], bool, int] = 'auto', daily_seasonality: Union[Literal['auto'], bool, int] = 'auto', daily_seasonality_glocal_mode: Union[Literal['auto'], bool, int] = 'auto', seasonality_mode: Literal['additive', 'multiplicative'] = 'additive', seasonality_reg: float = 0, season_global_local: Literal['global', 'local', 'glocal'] = 'global', seasonality_local_reg: Optional[Union[bool, float]] = False, future_regressors_model: Literal['linear', 'neural_nets', 'shared_neural_nets'] = 'linear', future_regressors_d_hidden: int = 4, future_regressors_num_hidden_layers: int = 2, n_forecasts: int = 1, n_lags: int = 0, ar_layers: Optional[list] = [], ar_reg: Optional[float] = None, lagged_reg_layers: Optional[list] = [], learning_rate: Optional[float] = None, epochs: Optional[int] = None, batch_size: Optional[int] = None, loss_func: Union[str, torch.nn.modules.loss._Loss, Callable] = 'SmoothL1Loss', optimizer: Union[str, Type[torch.optim.optimizer.Optimizer]] = 'AdamW', newer_samples_weight: float = 2, newer_samples_start: float = 0.0, quantiles: List[float] = [], impute_missing: bool = True, impute_linear: int = 10, impute_rolling: int = 10, drop_missing: bool = False, collect_metrics: Union[bool, list, dict] = True, normalize: Literal['auto', 'soft', 'soft1', 'minmax', 'standardize', 'off'] = 'auto', global_normalization: bool = False, global_time_normalization: bool = True, unknown_data_normalization: bool = False, accelerator: Optional[str] = None, trainer_config: dict = {}, prediction_frequency: Optional[dict] = None)

class neuralprophet.time_dataset.GlobalTimeDataset(df, **kwargs)
class neuralprophet.time_dataset.TimeDataset(df, name, **kwargs)
Create a PyTorch dataset of a tabularized time-series

drop_nan_after_init(df, predict_steps, drop_missing)
Checks if inputs/targets contain any NaN values and drops them, if user opts to. :param drop_missing: whether to automatically drop missing samples from the data :type drop_missing: bool :param predict_steps: number of steps to predict :type predict_steps: int

filter_samples_after_init(prediction_frequency=None)
Filters samples from the dataset based on the forecast frequency. :param prediction_frequency: periodic interval in which forecasts should be made. :type prediction_frequency: int :param Note: :param —-: :param E.g. if prediction_frequency=7: :param forecasts are only made on every 7th step (once in a week in case of daily: :param resolution).:

init_after_tabularized(inputs, targets=None)
Create Timedataset with data. :param inputs: Identical to returns from tabularize_univariate_datetime() :type inputs: ordered dict :param targets: Identical to returns from tabularize_univariate_datetime() :type targets: np.array, float

neuralprophet.time_dataset.fourier_series(dates, period, series_order)
Provides Fourier series components with the specified frequency and order. .. note:: Identical to OG Prophet.

Parameters
dates (pd.Series) – Containing timestamps

period (float) – Number of days of the period

series_order (int) – Number of fourier components

Returns
Matrix with seasonality features

Return type
np.array

neuralprophet.time_dataset.fourier_series_t(t, period, series_order)
Provides Fourier series components with the specified frequency and order. .. note:: This function is identical to Meta AI’s Prophet Library

Parameters
t (pd.Series, float) – Containing time as floating point number of days

period (float) – Number of days of the period

series_order (int) – Number of fourier components

Returns
Matrix with seasonality features

Return type
np.array

neuralprophet.time_dataset.make_country_specific_holidays_df(year_list, country)
Make dataframe of country specific holidays for given years and countries :param year_list: List of years :type year_list: list :param country: List of country names :type country: str, list

Returns
Containing country specific holidays df with columns ‘ds’ and ‘holiday’

Return type
pd.DataFrame

neuralprophet.time_dataset.make_events_features(df, config_events: Optional[OrderedDict[str, neuralprophet.configure.Event]] = None, config_country_holidays=None)
Construct arrays of all event features :param df: Dataframe with all values including the user specified events (provided by user) :type df: pd.DataFrame :param config_events: User specified events, each with their upper, lower windows (int), regularization :type config_events: configure.ConfigEvents :param config_country_holidays: Configurations (holiday_names, upper, lower windows, regularization) for country specific holidays :type config_country_holidays: configure.ConfigCountryHolidays

Returns
np.array – All additive event features (both user specified and country specific)

np.array – All multiplicative event features (both user specified and country specific)

neuralprophet.time_dataset.make_regressors_features(df, config_regressors)
Construct arrays of all scalar regressor features :param df: Dataframe with all values including the user specified regressors :type df: pd.DataFrame :param config_regressors: User specified regressors config :type config_regressors: configure.ConfigFutureRegressors

Returns
np.array – All additive regressor features

np.array – All multiplicative regressor features

neuralprophet.time_dataset.seasonal_features_from_dates(df, config_seasonality: neuralprophet.configure.ConfigSeasonality)
Dataframe with seasonality features. Includes seasonality features, holiday features, and added regressors. :param df: Dataframe with all values :type df: pd.DataFrame :param config_seasonality: Configuration for seasonalities :type config_seasonality: configure.ConfigSeasonality

Returns
Dictionary with keys for each period name containing an np.array with the respective regression features. each with dims: (len(dates), 2*fourier_order)

Return type
OrderedDict

neuralprophet.time_dataset.tabularize_univariate_datetime(df, predict_mode=False, n_lags=0, n_forecasts=1, predict_steps=1, config_seasonality: Optional[neuralprophet.configure.ConfigSeasonality] = None, config_events: Optional[OrderedDict[str, neuralprophet.configure.Event]] = None, config_country_holidays=None, config_lagged_regressors: Optional[OrderedDict[str, neuralprophet.configure.LaggedRegressor]] = None, config_regressors: Optional[neuralprophet.configure.ConfigFutureRegressors] = None, config_missing=None, prediction_frequency=None)
Create a tabular dataset from univariate timeseries for supervised forecasting. .. note:

Data must have no gaps.
If data contains missing values, they are ignored for the creation of the dataset.
Parameters
df (pd.DataFrame) – Sequence of observations with original ds, y and normalized t, y_scaled columns

config_seasonality (configure.ConfigSeasonality) – Configuration for seasonalities

n_lags (int) – Number of lagged values of series to include as model inputs (aka AR-order)

n_forecasts (int) – Number of steps to forecast into future

config_events (configure.ConfigEvents) – User specified events, each with their upper, lower windows (int) and regularization

config_country_holidays (configure.ConfigCountryHolidays) – Configurations (holiday_names, upper, lower windows, regularization) for country specific holidays

config_lagged_regressors (configure.ConfigLaggedRegressors) – Configurations for lagged regressors

config_regressors (configure.ConfigFutureRegressors) – Configuration for regressors

predict_mode (bool) –

Chooses the prediction mode Options

(default) False: Includes target values

True: Does not include targets but includes entire dataset as input

Returns
OrderedDict – Model inputs, each of len(df) but with varying dimensions .. note:

Contains the following data:
Model Inputs
    * ``time`` (np.array, float), dims: (num_samples, 1)
    * ``seasonalities`` (OrderedDict), named seasonalities
    each with features (np.array, float) - dims: (num_samples, n_features[name])
    * ``lags`` (np.array, float), dims: (num_samples, n_lags)
    * ``covariates`` (OrderedDict), named covariates,
    each with features (np.array, float) of dims: (num_samples, n_lags)
    * ``events`` (OrderedDict), events,
    each with features (np.array, float) of dims: (num_samples, n_lags)
    * ``regressors`` (OrderedDict), regressors,
    each with features (np.array, float) of dims: (num_samples, n_lags)
np.array, float – Targets to be predicted of same length as each of the model inputs, dims: (num_samples, n_forecasts)


Core Module Documentation
Copyright © 2024, Oskar Triebe

Core Module Documentation
class neuralprophet.time_net.DeepNet(d_inputs, d_outputs, lagged_reg_layers=[])
A simple, general purpose, fully connected network

forward(x)
This method defines the network layering and activation functions

class neuralprophet.time_net.FlatNet(d_inputs, d_outputs)
Linear regression fun

forward(x)
Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.

class neuralprophet.time_net.TimeNet(config_seasonality: neuralprophet.configure.ConfigSeasonality, config_train: Optional[neuralprophet.configure.Train] = None, config_trend: Optional[neuralprophet.configure.Trend] = None, config_ar: Optional[neuralprophet.configure.AR] = None, config_normalization: Optional[neuralprophet.configure.Normalization] = None, config_lagged_regressors: Optional[OrderedDict[str, neuralprophet.configure.LaggedRegressor]] = None, config_regressors: Optional[neuralprophet.configure.ConfigFutureRegressors] = None, config_events: Optional[OrderedDict[str, neuralprophet.configure.Event]] = None, config_holidays: Optional[neuralprophet.configure.Holidays] = None, n_forecasts: int = 1, n_lags: int = 0, max_lags: int = 0, ar_layers: Optional[List[int]] = [], lagged_reg_layers: Optional[List[int]] = [], compute_components_flag: bool = False, metrics: Optional[Union[Dict, bool]] = {}, id_list: List[str] = ['__df__'], num_trends_modelled: int = 1, num_seasonalities_modelled: int = 1, num_seasonalities_modelled_dict: Optional[dict] = None, meta_used_in_model: bool = False)
Linear time regression fun and some not so linear fun. A modular model that models classic time-series components

trend

seasonality

auto-regression (as AR-Net)

covariates (as AR-Net)

apriori regressors

events and holidays

by using Neural Network components. The Auto-regression and covariate components can be configured as a deeper network (AR-Net).

property ar_weights: torch.Tensor
sets property auto-regression weights for regularization. Update if AR is modelled differently

auto_regression(lags: Union[torch.Tensor, float]) → torch.Tensor
Computes auto-regessive model component AR-Net. :param lags: Previous times series values, dims: (batch, n_lags) :type lags: torch.Tensor, float

Returns
Forecast component of dims: (batch, n_forecasts)

Return type
torch.Tensor

compute_components(inputs: Dict, components_raw: Dict, meta: Dict) → Dict
This method returns the values of each model component. .. note:: Time input is required. Minimum model setup is a linear trend.

Parameters
inputs (dict) –

Model inputs, each of len(df) but with varying dimensions .. note:

Contains the following data:
Model Inputs
    * ``time`` (torch.Tensor , loat), normalized time, dims: (batch, n_forecasts)
    * ``lags`` (torch.Tensor, float), dims: (batch, n_lags)
    * ``seasonalities`` (torch.Tensor, float), dict of named seasonalities (keys) with their features
    (values), dims of each dict value (batch, n_forecasts, n_features)
    * ``covariates`` (torch.Tensor, float), dict of named covariates (keys) with their features
    (values), dims of each dict value: (batch, n_lags)
    * ``events`` (torch.Tensor, float), all event features, dims (batch, n_forecasts, n_features)
    * ``regressors``(torch.Tensor, float), all regressor features, dims (batch, n_forecasts, n_features)
components_raw (dict) – components to be computed

dict
Containing forecast coomponents with elements of dims (batch, n_forecasts)

configure_optimizers()
Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you’d need one. But in the case of GANs or similar you might have multiple.

Returns
Any of these 6 options.

Single optimizer.

List or Tuple of optimizers.

Two lists - The first list has multiple optimizers, and the second has multiple LR schedulers (or multiple lr_scheduler_config).

Dictionary, with an "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_scheduler_config.

Tuple of dictionaries as described above, with an optional "frequency" key.

None - Fit will run without any optimizer.

The lr_scheduler_config is a dictionary which contains the scheduler and its associated configuration. The default configuration is shown below.

lr_scheduler_config = {
    # REQUIRED: The scheduler instance
    "scheduler": lr_scheduler,
    # The unit of the scheduler's step size, could also be 'step'.
    # 'epoch' updates the scheduler on epoch end whereas 'step'
    # updates it after a optimizer update.
    "interval": "epoch",
    # How many epochs/steps should pass between calls to
    # `scheduler.step()`. 1 corresponds to updating the learning
    # rate after every epoch/step.
    "frequency": 1,
    # Metric to to monitor for schedulers like `ReduceLROnPlateau`
    "monitor": "val_loss",
    # If set to `True`, will enforce that the value specified 'monitor'
    # is available when the scheduler is updated, thus stopping
    # training if not found. If set to `False`, it will only produce a warning
    "strict": True,
    # If using the `LearningRateMonitor` callback to monitor the
    # learning rate progress, this keyword can be used to specify
    # a custom logged name
    "name": None,
}
When there are schedulers in which the .step() method is conditioned on a value, such as the torch.optim.lr_scheduler.ReduceLROnPlateau scheduler, Lightning requires that the lr_scheduler_config contains the keyword "monitor" set to the metric name that the scheduler should be conditioned on.

Metrics can be made available to monitor by simply logging it using self.log('metric_to_track', metric_val) in your LightningModule.

Note

The frequency value specified in a dict along with the optimizer key is an int corresponding to the number of sequential batches optimized with the specific optimizer. It should be given to none or to all of the optimizers. There is a difference between passing multiple optimizers in a list, and passing multiple optimizers in dictionaries with a frequency of 1:

In the former case, all optimizers will operate on the given batch in each optimization step.

In the latter, only one optimizer will operate on the given batch at every step.

This is different from the frequency value specified in the lr_scheduler_config mentioned above.

def configure_optimizers(self):
    optimizer_one = torch.optim.SGD(self.model.parameters(), lr=0.01)
    optimizer_two = torch.optim.SGD(self.model.parameters(), lr=0.01)
    return [
        {"optimizer": optimizer_one, "frequency": 5},
        {"optimizer": optimizer_two, "frequency": 10},
    ]
In this example, the first optimizer will be used for the first 5 steps, the second optimizer for the next 10 steps and that cycle will continue. If an LR scheduler is specified for an optimizer using the lr_scheduler key in the above dict, the scheduler will only be updated when its optimizer is being used.

Examples:

# most cases. no learning rate scheduler
def configure_optimizers(self):
    return Adam(self.parameters(), lr=1e-3)

# multiple optimizer case (e.g.: GAN)
def configure_optimizers(self):
    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
    dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
    return gen_opt, dis_opt

# example with learning rate schedulers
def configure_optimizers(self):
    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
    dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
    dis_sch = CosineAnnealing(dis_opt, T_max=10)
    return [gen_opt, dis_opt], [dis_sch]

# example with step-based learning rate schedulers
# each optimizer has its own scheduler
def configure_optimizers(self):
    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
    dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
    gen_sch = {
        'scheduler': ExponentialLR(gen_opt, 0.99),
        'interval': 'step'  # called after each training step
    }
    dis_sch = CosineAnnealing(dis_opt, T_max=10) # called every epoch
    return [gen_opt, dis_opt], [gen_sch, dis_sch]

# example with optimizer frequencies
# see training procedure in `Improved Training of Wasserstein GANs`, Algorithm 1
# https://arxiv.org/abs/1704.00028
def configure_optimizers(self):
    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
    dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
    n_critic = 5
    return (
        {'optimizer': dis_opt, 'frequency': n_critic},
        {'optimizer': gen_opt, 'frequency': 1}
    )
Note

Some things to know:

Lightning calls .backward() and .step() on each optimizer as needed.

If learning rate scheduler is specified in configure_optimizers() with key "interval" (default “epoch”) in the scheduler configuration, Lightning will call the scheduler’s .step() method automatically in case of automatic optimization.

If you use 16-bit precision (precision=16), Lightning will automatically handle the optimizers.

If you use multiple optimizers, training_step() will have an additional optimizer_idx parameter.

If you use torch.optim.LBFGS, Lightning handles the closure function automatically for you.

If you use multiple optimizers, gradients will be calculated only for the parameters of current optimizer at each training step.

If you need to control how often those optimizers step or override the default .step() schedule, override the optimizer_step() hook.

denormalize(ts)
Denormalize timeseries :param target: ts tensor :type target: torch.Tensor

Return type
denormalized timeseries

forward(inputs: Dict, meta: Optional[Dict] = None, compute_components_flag: bool = False) → torch.Tensor
This method defines the model forward pass. .. note:: Time input is required. Minimum model setup is a linear trend.

Parameters
inputs (dict) –

Model inputs, each of len(df) but with varying dimensions .. note:

Contains the following data:
Model Inputs
    * ``time`` (torch.Tensor , loat), normalized time, dims: (batch, n_forecasts)
    * ``lags`` (torch.Tensor, float), dims: (batch, n_lags)
    * ``seasonalities`` (torch.Tensor, float), dict of named seasonalities (keys) with their features
    (values), dims of each dict value (batch, n_forecasts, n_features)
    * ``covariates`` (torch.Tensor, float), dict of named covariates (keys) with their features
    (values), dims of each dict value: (batch, n_lags)
    * ``events`` (torch.Tensor, float), all event features, dims (batch, n_forecasts, n_features)
    * ``regressors``(torch.Tensor, float), all regressor features, dims (batch, n_forecasts, n_features)
    * ``predict_mode`` (bool), optional and only passed during prediction
meta (dict, default=None) –

Metadata about the all the samples of the model input batch. Contains the following: Model Meta:

df_name (list, str), time series ID corresponding to each sample of the input batch.

The default None value allows the forward method to be used without providing the meta argument. This was designed to avoid issues with the library lr_finder https://github.com/davidtvs/pytorch-lr-finder while having config_trend.trend_global_local="local". The turnaround consists on passing the same meta (dummy ID) to all the samples of the batch. Internally, this is equivalent to use config_trend.trend_global_local="global" to find the optimal learning rate.

compute_components_flag (bool, default=False) – If True, components will be computed.

Returns
Forecast of dims (batch, n_forecasts, no_quantiles)

Return type
torch.Tensor

forward_covar_net(covariates)
Compute all covariate components. :param covariates: dict of named covariates (keys) with their features (values)

dims of each dict value: (batch, n_lags)

Returns
Forecast component of dims (batch, n_forecasts, quantiles)

Return type
torch.Tensor

get_covar_weights(covar_input=None) → torch.Tensor
Get attributions of covariates network w.r.t. the model input.

get_event_weights(name: str) → Dict[str, torch.Tensor]
Retrieve the weights of event features given the name :param name: Event name :type name: str

Returns
Dict of the weights of all offsets corresponding to a particular event

Return type
OrderedDict

predict_step(batch, batch_idx, dataloader_idx=0)
Step function called during predict(). By default, it calls forward(). Override to add any processing logic.

The predict_step() is used to scale inference on multi-devices.

To prevent an OOM error, it is possible to use BasePredictionWriter callback to write the predictions to disk or database after each batch or on epoch end.

The BasePredictionWriter should be used while using a spawn based accelerator. This happens for Trainer(strategy="ddp_spawn") or training on 8 TPU cores with Trainer(accelerator="tpu", devices=8) as predictions won’t be returned.

Example

class MyModel(LightningModule):

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

dm = ...
model = MyModel()
trainer = Trainer(accelerator="gpu", devices=2)
predictions = trainer.predict(model, dm)
Parameters
batch – Current batch.

batch_idx – Index of current batch.

dataloader_idx – Index of the current dataloader.

Returns
Predicted output

scalar_features_effects(features: torch.Tensor, params: torch.nn.parameter.Parameter, indices=None) → torch.Tensor
Computes events component of the model :param features: Features (either additive or multiplicative) related to event component dims (batch, n_forecasts,

n_features)

Parameters
params (nn.Parameter) – Params (either additive or multiplicative) related to events

indices (list of int) – Indices in the feature tensors related to a particular event

Returns
Forecast component of dims (batch, n_forecasts)

Return type
torch.Tensor

set_covar_weights(covar_weights: torch.Tensor)
Function to set the covariate weights for later interpretation in compute_components. This function is needed since the gradient information is not available during the predict_step method and attributions cannot be calculated in compute_components.

Parameters
covar_weights (torch.Tensor) – _description_

test_step(batch, batch_idx)
Operates on a single batch of data from the test set. In this step you’d normally generate examples or calculate anything of interest such as accuracy.

# the pseudocode for these calls
test_outs = []
for test_batch in test_data:
    out = test_step(test_batch)
    test_outs.append(out)
test_epoch_end(test_outs)
Parameters
batch – The output of your DataLoader.

batch_idx – The index of this batch.

dataloader_id – The index of the dataloader that produced this batch. (only if multiple test dataloaders used).

Returns
Any of.

Any object or value

None - Testing will skip to the next batch

# if you have one test dataloader:
def test_step(self, batch, batch_idx):
    ...


# if you have multiple test dataloaders:
def test_step(self, batch, batch_idx, dataloader_idx=0):
    ...
Examples:

# CASE 1: A single test dataset
def test_step(self, batch, batch_idx):
    x, y = batch

    # implement your own
    out = self(x)
    loss = self.loss(out, y)

    # log 6 example images
    # or generated text... or whatever
    sample_imgs = x[:6]
    grid = torchvision.utils.make_grid(sample_imgs)
    self.logger.experiment.add_image('example_images', grid, 0)

    # calculate acc
    labels_hat = torch.argmax(out, dim=1)
    test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

    # log the outputs!
    self.log_dict({'test_loss': loss, 'test_acc': test_acc})
If you pass in multiple test dataloaders, test_step() will have an additional argument. We recommend setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

# CASE 2: multiple test dataloaders
def test_step(self, batch, batch_idx, dataloader_idx=0):
    # dataloader_idx tells you which dataset this is.
    ...
Note

If you don’t need to test you don’t need to implement this method.

Note

When the test_step() is called, the model has been put in eval mode and PyTorch gradients have been disabled. At the end of the test epoch, the model goes back to training mode and gradients are enabled.

training_step(batch, batch_idx)
Here you compute and return the training loss and some additional metrics for e.g. the progress bar or logger.

Parameters
batch (Tensor | (Tensor, …) | [Tensor, …]) – The output of your DataLoader. A tensor, tuple or list.

batch_idx (int) – Integer displaying index of this batch

optimizer_idx (int) – When using multiple optimizers, this argument will also be present.

hiddens (Any) – Passed in if :paramref:`~pytorch_lightning.core.module.LightningModule.truncated_bptt_steps` > 0.

Returns
Any of.

Tensor - The loss tensor

dict - A dictionary. Can include any keys, but must include the key 'loss'

None - Training will skip to the next batch. This is only for automatic optimization.
This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

In this step you’d normally do the forward pass and calculate the loss for a batch. You can also do fancier things like multiple forward passes or something model specific.

Example:

def training_step(self, batch, batch_idx):
    x, y, z = batch
    out = self.encoder(x)
    loss = self.loss(out, x)
    return loss
If you define multiple optimizers, this step will be called with an additional optimizer_idx parameter.

# Multiple optimizers (e.g.: GANs)
def training_step(self, batch, batch_idx, optimizer_idx):
    if optimizer_idx == 0:
        # do training_step with encoder
        ...
    if optimizer_idx == 1:
        # do training_step with decoder
        ...
If you add truncated back propagation through time you will also get an additional argument with the hidden states of the previous step.

# Truncated back-propagation through time
def training_step(self, batch, batch_idx, hiddens):
    # hiddens are the hidden states from the previous truncated backprop step
    out, hiddens = self.lstm(data, hiddens)
    loss = ...
    return {"loss": loss, "hiddens": hiddens}
Note

The loss value shown in the progress bar is smoothed (averaged) over the last values, so it differs from the actual loss returned in train/validation step.

Note

When accumulate_grad_batches > 1, the loss returned here will be automatically normalized by accumulate_grad_batches internally.

validation_step(batch, batch_idx)
Operates on a single batch of data from the validation set. In this step you’d might generate examples or calculate anything of interest like accuracy.

# the pseudocode for these calls
val_outs = []
for val_batch in val_data:
    out = validation_step(val_batch)
    val_outs.append(out)
validation_epoch_end(val_outs)
Parameters
batch – The output of your DataLoader.

batch_idx – The index of this batch.

dataloader_idx – The index of the dataloader that produced this batch. (only if multiple val dataloaders used)

Returns
Any object or value

None - Validation will skip to the next batch

# pseudocode of order
val_outs = []
for val_batch in val_data:
    out = validation_step(val_batch)
    if defined("validation_step_end"):
        out = validation_step_end(out)
    val_outs.append(out)
val_outs = validation_epoch_end(val_outs)
# if you have one val dataloader:
def validation_step(self, batch, batch_idx):
    ...


# if you have multiple val dataloaders:
def validation_step(self, batch, batch_idx, dataloader_idx=0):
    ...
Examples:

# CASE 1: A single validation dataset
def validation_step(self, batch, batch_idx):
    x, y = batch

    # implement your own
    out = self(x)
    loss = self.loss(out, y)

    # log 6 example images
    # or generated text... or whatever
    sample_imgs = x[:6]
    grid = torchvision.utils.make_grid(sample_imgs)
    self.logger.experiment.add_image('example_images', grid, 0)

    # calculate acc
    labels_hat = torch.argmax(out, dim=1)
    val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

    # log the outputs!
    self.log_dict({'val_loss': loss, 'val_acc': val_acc})
If you pass in multiple val dataloaders, validation_step() will have an additional argument. We recommend setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

# CASE 2: multiple validation dataloaders
def validation_step(self, batch, batch_idx, dataloader_idx=0):
    # dataloader_idx tells you which dataset this is.
    ...
Note

If you don’t need to validate you don’t need to implement this method.

Note

When the validation_step() is called, the model has been put in eval mode and PyTorch gradients have been disabled. At the end of validation, the model goes back to training mode and gradients are enabled.