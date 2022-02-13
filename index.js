const fs = require("fs");
const express = require("express");
const dotenv = require("dotenv");
const mongoose = require("mongoose");
const multer = require("multer");
const slugify = require("slugify");
const User = require("./models/User");

dotenv.config({ path: './config.env' });

mongoose.connect(process.env.DB_URI, (error) => {
    if(error) {
        console.log("ERROR CONNECTING TO MONGODB ", error);
    } else {
        console.log("DB CONNECTION SUCCESSFUL");
    }
});

const app = express();
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const fs = require("fs");
        const slug = slugify(req.body.name, { trim: true, lower: true });
        let uploadPath = "";
        if(req.body.uploadPath) {
            uploadPath = req.body.uploadPath;
        } else {
            uploadPath = "./images/" + slug + '-' + Date.now();
            if(!fs.existsSync(uploadPath)) {
                fs.mkdirSync(uploadPath);
                uploadPath += "/signatures";
                fs.mkdirSync(uploadPath);
            }
            req.body.uploadPath = uploadPath;
        }
        cb(null, uploadPath);
    },
    filename: function (req, file, cb) {
        const fileName = `${file.originalname.split(".")[0]}-${Date.now()}.${file.mimetype.split("/")[1]}`;
        cb(null, fileName);
    }
});
const testImageStorage = multer.diskStorage({
    destination: function(req, file, cb) {
        let uploadPath = './images/test';
        cb(null, uploadPath);
    },
    filename: function(req, file, cb) {
        let fileName = `${file.originalname.split(".")[0]}-${Date.now()}.${file.mimetype.split("/")[1]}`;
        cb(null, fileName);
    }
})
const upload = multer({
    storage
});
const uploadTest = multer({
    storage: testImageStorage
})

app.use(express.json());
app.use(express.urlencoded());   
app.use(express.static('./public'));
app.use("/static", express.static('./images'));
app.set('view engine', 'pug');

app.post("/run-script", (req, res, next) => {
    const trainPath = req.body.trainPath;
    const name = req.body.name;
    const { spawn } = require('child_process');
    const pyProg = spawn('python', ['./scripts/train.py', trainPath, name, "train"]);

    pyProg.stdout.on('data', async function(data) {
        console.log("LOGGING DIRECTLY FROM SCRIPT ", data.toString('utf-8'));
        let responseData = data.toString('utf-8');
        if(responseData.includes("Model-Summary: ")) {
            responseData = responseData.split("Model-Summary: ")[1];
            let jsonData = JSON.parse(responseData);
            await User.findOneAndUpdate({ name: name, trainPath: trainPath}, {
                accuracy: jsonData.accuracy,
                modelSavePath: jsonData.model_save_path
            });
        }
    });

    pyProg.stdout.on('end', function(data) {
        res.end('Training completed!');
    });

    pyProg.stderr.on('data', function(data) {
        console.log('stderr: Error is ' + data.toString('utf-8'));
    });
});

app.post("/test", uploadTest.single('image'), (req, res) => {
    const modelSavePath = req.body.modelSavePath;
    const testPath = "./" + req.file.path;
    const { spawn } = require('child_process');
    const pyProg = spawn('python', ['./scripts/train.py', testPath, modelSavePath, "test"]);

    pyProg.stdout.on('data', async function(data) {
        console.log("LOGGING DIRECTLY FROM SCRIPT ", data.toString('utf-8'));
        let responseData = data.toString('utf-8');
        if(responseData.includes("Test-Summary: ")) {
            responseData = responseData.split("Test-Summary: ")[1];
            let jsonData = JSON.parse(responseData);
            const user = await User.findOne({modelSavePath: modelSavePath});
            let newTestResult = { testPath: testPath, testAccuracy: jsonData.test_accuracy };
            if(!user.testResults) {
                user.testResults = [];
            }
            user.testResults.push(newTestResult);
            await user.save();
        }
    });

    pyProg.stdout.on('end', function(data) {
        res.end('Testing completed!');
    });

    pyProg.stderr.on('data', function(data) {
        console.log('stderr: Error is ' + data.toString('utf-8'));
    });
});

app.get("/", async (req, res) => {
    const users = await User.find();
    res.render("index", { url: "/", users });
});

app.get("/add", (req, res) => {
    res.render("add", {url: "/add" });
});

app.get("/user/:id", async (req, res) => {
    const id = req.params.id;
    const user = await User.findById(id);
    fs.readdir(user.uploadPath, (err, files) => {
        let imageList = []; 
        files.forEach(file => {
            if(!file.toLowerCase().startsWith("aug-")) {
                imageList.push(file);
            }
        });
        res.render("train_test", { user, imageList })
    });
})

app.post("/add", upload.array("images", 9), async (req, res) => {
    try {
        await User.create({
            name: req.body.name,
            uploadPath: req.body.uploadPath,
            augment: req.body.augment == "on"? true: false
        });
        res.render("add", { url: "/add", successMessage: "User was created successfully." })
    } catch (error) {
        const errrorMessage = error.message;
        res.render("add", { url: "/add", errorMessage: errrorMessage });
    }
});

app.listen(process.env.PORT, () => {
    console.log(`Listening at port ${process.env.PORT}`)
});