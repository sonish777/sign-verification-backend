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
const upload = multer({
    storage
});

app.use(express.json());
app.use(express.urlencoded());   
app.use(express.static('./public'));
app.use("/static", express.static('./images'));
app.set('view engine', 'pug');

app.get("/run-script", (req, res, next) => {
    const { spawn } = require('child_process');
    const pyProg = spawn('python', ['./scripts/train.py', "images/binita"]);

    pyProg.stdout.on('data', function(data) {
        console.log(data.toString('utf-8'));
    });

    pyProg.stdout.on('end', function(data) {
        console.log("DATA IS", data);
        res.end('Training completed!');
    });
});

app.get("/", async (req, res) => {
    const users = await User.find();
    res.render("index", { url: "/", users });
});

app.get("/add", (req, res) => {
    res.render("add", {url: "/add" });
});

app.get("/:id", async (req, res) => {
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