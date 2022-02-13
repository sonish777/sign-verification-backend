const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
    name: { type: String, required: true },
    uploadPath: { type: String },
    modelSavePath: { type: String },
    augment: { type: Boolean, default: true },
    accuracy: { type: Number },
    testResults: [
        {
            testPath: {type: String},
            testAccuracy: {type: Number}
        }
    ]
});

const User = mongoose.model('User', userSchema);

module.exports = User;