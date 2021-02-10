// import * as tf from '@tensorflow/tfjs@3.0.0';

const CANVAS_SIZE = 280;
const CANVAS_SCALE = 1.0;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const option = document.getElementById("imageSelect");

var fashion = ["T-shirt", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Angkle boot"
];

// 모델 로드
async function load_model() {
    m = tf.loadLayersModel('../fmodel/model.json');
    return m;
}
loadingModelPromise = load_model();

ctx.fillStyle = "#212121";
ctx.font = "28px sans-serif";
ctx.textAlign = "center";
ctx.textBaseLine = "middle";
ctx.fillText("로딩중...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

// 이미지 바꾸기
function imageKindChange() {
    img_source = option.value;
    var img = new Image();

    img.addEventListener('load', function() {
        ctx.drawImage(img, 0, 0);
        updatePrediction();
    }, false);
    img.src = img_source;
}

// 예측 함수
async function updatePrediction() {
    const model = await tf.loadLayersModel("../fmodel/model.json");

    const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const data = imgData.data;
    var arr = [];
    // 그레이 스케일링
    for (var i = 0; i < data.length; i += 4) {
        var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        arr.push(data[i]);
    }
    // 이미지 리사이징
    arr = arr.map(x => x / 255); // Normalization
    arr = tf.tensor3d(arr, [280, 280, 1]);
    let img = tf.image.resizeBilinear(arr, [28, 28]);
    img = tf.expandDims(img, 0);
    const prediction = model.predict(img);
    var result = prediction.argMax(1).dataSync();
    var r = result[0];
    document.getElementById("result").innerHTML = fashion[r];
}

loadingModelPromise.then(() => {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.font = '20px sans-serif';
    ctx.fillText("이미지를 선택해주세요!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
})