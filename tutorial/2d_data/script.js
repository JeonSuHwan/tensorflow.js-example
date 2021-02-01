console.log('Hello Tensorflow');

/*
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
// 입력할 데이터를 로드
async function getData() {
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();
  const cleaned = carsData.map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));

  return cleaned;
}

// 형식 지정 및 시각화
async function run() {
  // 학습할 데이터를 로드하고 시각화함
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot({
    name: 'Horsepower v MPG'
  }, {
    values
  }, {
    xLabel: 'Horsepower',
    yLabel: 'MPG',
    height: 300
  });

  // 모델 생성
  const model = createModel();
  tfvis.show.modelSummary({
    name: 'Model Summary'
  }, model);

  // 훈련할 데이터 변환
  const tensorData = convertToTensor(data);
  const {
    inputs,
    labels
  } = tensorData;

  // 모델 학습
  await trainModel(model, inputs, labels);
  console.log('Done Training');

  // 모델 예측
  testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);

// 모델 아키텍쳐 정의
function createModel() {
  // 시퀀스 모델 생성
  const model = tf.sequential();

  // 인풋 레이어 추가
  model.add(tf.layers.dense({
    inputShape: [1],
    units: 1,
    useBias: true
  }));

  model.add(tf.layers.dense({
    units: 50,
    activation: 'sigmoid'
  }));

  // 아웃풋 레이어 추가
  model.add(tf.layers.dense({
    units: 1,
    useBias: true
  }));

  return model;
}

// 훈련 데이터 준비
/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practice of _shuffling_
 * the data and _normalizing_ the getData
 * MPG on the y-axis.
 */

function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. 데이터 셔플
    tf.util.shuffle(data);

    // Step 2. 데이터를 텐서로 변경
    const inputs = data.map(d => d.horsepower);
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Step 3. 데이터 정규화 (Min-Max scaling사용)
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
  batchSize,
  epochs,
  callbacks: tfvis.show.fitCallbacks(
    { name: 'Training Performance' },
    ['loss', 'mse'],
    { height: 200, callbacks: ['onEpochEnd'] }
  )
});
}

function testModel(model, inputData, normalizationData){
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  const [xs, preds] = tf.tidy(()=>{
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs
    .mul(inputMax.sub(inputMin))
    .add(inputMin);

    const unNormPreds = preds
    .mul(labelMax.sub(labelMin))
    .add(labelMin);

    // Un-normalize the getData
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'Horsepwer',
      yLabel: 'MPG',
      height: 300
    }
  );
}
