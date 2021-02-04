import { MnistData } from './data.js';

// 모델 정의
function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 3,
        filters: 16,
        strides: 1,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'heUniform'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        strides: 1,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'heUniform'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    model.add(tf.layers.dropout({ rate: 0.25 }));

    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        strides: 1,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'heUniform'
    }))

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [1, 1] }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
        useBias: true,
        kernelInitializer: 'heUniform'
    }));

    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax',
        kernelInitializer: 'heUniform'
    }));

    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

// 모델 학습
async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training',
        tab: 'Model',
        styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 15,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

// 모델 평가
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds, labels];
}

async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    console.log([preds, labels]);
    labels.dispose();
}

async function run() {
    const data = new MnistData();
    await data.load();

    const model = getModel();

    console.log('Train start!');
    await train(model, data);
    console.log('Train finished');

    await showAccuracy(model, data);

    // 모델 저장
    await model.save('downloads://my-model');
    console.log('Save model complete!');
}

document.addEventListener('DOMContentLoaded', run);