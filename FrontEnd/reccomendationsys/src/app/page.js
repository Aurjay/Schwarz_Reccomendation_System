'use client'; 

import { useState, useEffect } from 'react';

const ProductRecommendation = () => {
  const [productMap, setProductMap] = useState({});
  const [selectedProduct1, setSelectedProduct1] = useState('');
  const [selectedProduct2, setSelectedProduct2] = useState('');
  const [loading, setLoading] = useState(false);
  const [recommendation, setRecommendation] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const productCategories = {
      17: 'baking ingredients', 91: 'soy lactosefree', 36: 'butter', 83: 'fresh vegetables',
      120: 'yogurt', 59: 'canned meals beans', 35: 'poultry counter', 37: 'ice cream ice',
      24: 'fresh fruits', 84: 'milk', 21: 'packaged cheese', 112: 'bread', 94: 'tea',
      8: 'bakery desserts', 52: 'frozen breakfast', 121: 'cereal', 86: 'eggs', 43: 'buns rolls',
      // (Shortened for brevity)
    };
    setProductMap(productCategories);
  }, []);

  const sendPostRequest = async () => {
    if (!selectedProduct1 || !selectedProduct2) {
      alert('Please select two products!');
      return;
    }

    setLoading(true);
    setRecommendation([]);
    setError('');

    try {
      const product1Id = parseInt(selectedProduct1);
      const product2Id = parseInt(selectedProduct2);

      const response = await fetch('https://flask-recommender-app-930534651361.us-central1.run.app/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          current_product_ids: [product1Id, product2Id],
        }),
      });

      if (!response.ok) throw new Error('Failed to fetch recommendations');

      const data = await response.json();
      setLoading(false);
      setRecommendation(data.recommended_products || []);
    } catch (err) {
      console.error('Error during POST request:', err);
      setLoading(false);
      setError('An error occurred while fetching the recommendation.');
    }
  };

  return (
    <div className="container">
 <h1>Product Recommendation system using Neural Coloborative Filtering</h1>
      <br />
      <h2>Based on your product selection the AI system in the backend recommends the product that the user is most likely to buy next.</h2>
      <div className="form-container">
        <div className="form-card">
          <label>Select Product 1:</label>
          <select
            value={selectedProduct1}
            onChange={(e) => setSelectedProduct1(e.target.value)}
          >
            <option value="">-- Select a Product --</option>
            {Object.entries(productMap).map(([key, value]) => (
              <option key={key} value={key}>
                {value}
              </option>
            ))}
          </select>
        </div>

        <div className="form-card">
          <label>Select Product 2:</label>
          <select
            value={selectedProduct2}
            onChange={(e) => setSelectedProduct2(e.target.value)}
          >
            <option value="">-- Select a Product --</option>
            {Object.entries(productMap).map(([key, value]) => (
              <option key={key} value={key}>
                {value}
              </option>
            ))}
          </select>
        </div>
      </div>

      <button onClick={sendPostRequest} className={`button ${loading ? 'loading' : ''}`}>
        {loading ? 'Loading...' : 'Get Recommendations'}
      </button>

      {recommendation.length > 0 && (
        <div className="response">
          <h3>Recommendations:</h3>
          <ul>
            {recommendation.map((product, index) => (
              <li key={index}>{product}</li>
            ))}
          </ul>
        </div>
      )}

      {error && <p className="error">{error}</p>}

      <div className="links">
        <a href="https://github.com/Aurjay/Schwarz_Reccomendation_System" target="_blank" rel="noopener noreferrer">
          GitHub Project
        </a>
        <a href="https://www.deepakraj.site/" target="_blank" rel="noopener noreferrer">
          Project by Deepak Raj
        </a>
      </div>

      <style jsx>{`
        .container {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 20px;
          font-family: Arial, sans-serif;
          text-align: center;
        }

        h1 {
          color: #4CAF50;
          margin-bottom: 10px;
        }

        h2 {
          color: #555;
          margin-bottom: 20px;
        }

        .form-container {
          display: flex;
          justify-content: center;
          gap: 20px;
          margin-bottom: 20px;
          width: 100%;
          max-width: 800px;
        }

        .form-card {
          display: flex;
          flex-direction: column;
          width: 250px;
        }

        label {
          margin-bottom: 10px;
          font-weight: bold;
          color: #333;
        }

        select {
          padding: 10px;
          border-radius: 5px;
          border: 1px solid #ccc;
          font-size: 16px;
          background-color: white;
          color: black;
          width: 100%;
          box-sizing: border-box;
        }

        .button {
          padding: 12px 20px;
          font-size: 18px;
          background-color: #4CAF50;
          color: white;
          border: none;
          border-radius: 5px;
          cursor: pointer;
        }

        .button:hover {
          background-color: #45a049;
        }

        .button.loading {
          background-color: #f44336;
        }

        .response {
          margin-top: 20px;
        }

        .response ul {
          list-style-type: none;
          padding: 0;
        }

        .links {
          position: fixed;
          bottom: 20px;
          right: 20px;
          display: flex;
          flex-direction: column;
          gap: 10px;
          text-align: right;
        }

        .links a {
          color: #4CAF50;
          text-decoration: none;
          font-weight: bold;
        }

        .links a:hover {
          color: #45a049;
        }

        .error {
          color: red;
          margin-top: 10px;
        }
      `}</style>
    </div>
  );
};

export default ProductRecommendation;
