-- db_schema.sql
CREATE DATABASE IF NOT EXISTS disease_db;
USE disease_db;


CREATE TABLE IF NOT EXISTS patients (
id INT AUTO_INCREMENT PRIMARY KEY,
name VARCHAR(255),
age INT,
gender VARCHAR(32),
symptoms_json JSON,
predicted_disease VARCHAR(255),
confidence FLOAT,
created_at DATETIME
);