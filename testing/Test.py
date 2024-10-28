# Entrada y normalización
input_layer = Input(shape=(32, 32, 3), name="matrix_input")

# Estructura de la red
x = BatchNormalization(axis=-1, name="normalization_layer")(input_layer)
flatten_layer = Flatten(name="flattened_input")(x)

# Capas ocultas con mayor capacidad y Dropout
dense_512 = Dense(512, activation='relu', name="dense_512")(flatten_layer)
dropout_512 = Dropout(0.2)(dense_512)
dense_256 = Dense(256, activation='relu', name="dense_256")(dropout_512)
dropout_256 = Dropout(0.2)(dense_256)
dense_128 = Dense(128, activation='relu', name="dense_128")(dropout_256)
dense_64 = Dense(64, activation='relu', name="dense_64")(dense_128)


# Fine-grain prediction branch (100 classes)
fine_output = Dense(100,
                           activation='softmax',
                           name='fine_output')(dense_64)

# Coarse-grain prediction branch (20 classes)
coarse_output = Dense(20,
                             activation='softmax',
                             name='coarse_output')(dense_64)

# Defino el modelo con dos salidas
myModel = Model(
    inputs                  = input_layer,
    #outputs                 = [fine_output, coarse_output]
    outputs                 = [fine_output]
)


# Print model summary
myModel.summary()


""" Total params: 1,752,368 (6.68 MB)
 Trainable params: 1,752,362 (6.68 MB)
 Non-trainable params: 6 (24.00 B)"""
 
 rlrop = ReduceLROnPlateau(
    monitor = "val_accuracy",
    factor = 0.2,
    patience = 5,
    verbose = 1,
    min_lr = 1e-6
)


rlrop2 = ReduceLROnPlateau(
    monitor = "accuracy",
    factor = 0.2,
    patience = 5,
    verbose = 1,
    min_lr = 1e-6
)

es = EarlyStopping(
    monitor = "val_accuracy",
    patience = 10,
    verbose = 1,
    restore_best_weights = True
)

es2 = EarlyStopping(
    monitor = "accuracy",
    patience = 10,
    verbose = 1,
    restore_best_weights = True
)

mc2 = ModelCheckpoint(
    "best_weights.weights.h5",
    monitor = "accuracy",
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,

)


mc = ModelCheckpoint(
    "best_weights.weights.h5",
    monitor = "val_accuracy",
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,

)



tb = TensorBoard(
    log_dir = "logs"
)

myModel.compile(
    optimizer               = Adam(learning_rate=1e-3),
    #loss                    = {'fine_output': 'sparse_categorical_crossentropy', 'coarse_output': 'sparse_categorical_crossentropy'},
    loss                    = {'fine_output': 'sparse_categorical_crossentropy'},
    loss_weights            = None,
    #metrics                 = {'fine_output': 'categorical_accuracy', 'coarse_output': 'categorical_accuracy'},
    metrics                 = {'fine_output': 'accuracy'},
    weighted_metrics        = None,
    run_eagerly             = False,
    steps_per_execution     = 1,
    jit_compile             = "auto",
    auto_scale_loss         = True,
)
#FIT PRUEBA
history = myModel.fit(
    x                       = x_train_dup,
    y                       = {'fine_output': y_train_fine_dup},
    batch_size              = 128,
    epochs                  = 50,
    verbose                 = "auto",
    callbacks               = [LambdaCallback(on_epoch_end=lambda epoch, logs: print(logs if logs is not None else "No logs available")),rlrop,es,mc2,mc],
    validation_split        = 0.0,
    validation_data         = (x_test_dup, {'fine_output': y_test_fine_dup}),
    shuffle                 = True,
    class_weight            = None,
    sample_weight           = None,
    initial_epoch           = 0,
    steps_per_epoch         = None,
    validation_steps        = None,
    validation_batch_size   = None,
    validation_freq         = 1,
)