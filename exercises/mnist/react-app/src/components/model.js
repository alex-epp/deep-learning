export class Predictor {
    constructor() {
        this.predict_from_image = this.predict_from_image.bind(this);
    }
    predict_from_image(img) {
        return new Promise((resolve, reject) => {
            var a = new Float32Array(img.width*img.height);
            for(var i = 0; i < img.width*img.height*4; i += 4) {
                a[i/4] = (img.data[i] + img.data[i+1] + img.data[i+2]) / 3 / 255;
            }
            var xhttp = new XMLHttpRequest();
            xhttp.open("POST", "/predict", true);
            xhttp.setRequestHeader("Content-Type", "application/json");
            xhttp.onload = function() {
                resolve(xhttp.responseText);
            };
            xhttp.send(JSON.stringify(a));
        });
    }
}
