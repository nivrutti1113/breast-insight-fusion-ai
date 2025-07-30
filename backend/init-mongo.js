// MongoDB initialization script for breast cancer detection app

db = db.getSiblingDB('breast_cancer_detection');

// Create collections
db.createCollection('prediction_history');

// Create indexes for better performance
db.prediction_history.createIndex({ "created_at": -1 });
db.prediction_history.createIndex({ "prediction.classification": 1 });
db.prediction_history.createIndex({ "patient_id": 1 });

// Insert sample data (optional)
db.prediction_history.insertOne({
  prediction: {
    probability: 0.15,
    classification: "Benign",
    confidence: 0.70
  },
  image_metadata: {
    filename: "sample_mammogram.jpg",
    size: "512x512",
    content_type: "image/jpeg",
    file_size: 45678
  },
  model_info: {
    version: "1.0.0",
    architecture: "CNN with Transfer Learning",
    training_data: "Mammography dataset"
  },
  created_at: new Date(),
  updated_at: new Date(),
  notes: "Sample prediction for testing purposes"
});

print('Database initialized successfully!');
print('Collections created: prediction_history');
print('Indexes created for: created_at, classification, patient_id');
print('Sample data inserted: 1 prediction record');