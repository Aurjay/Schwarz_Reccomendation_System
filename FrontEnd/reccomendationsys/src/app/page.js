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
      53: 'cream', 115: 'water seltzer sparkling water', 110: 'pickled goods olives', 
      49: 'packaged poultry', 108: 'other creams cheeses', 29: 'honeys syrups nectars', 
      26: 'coffee', 31: 'refrigerated', 3: 'energy granola bars', 77: 'soft drinks', 
      30: 'latino foods', 111: 'plates bowls cups flatware', 54: 'paper goods', 20: 'oral hygiene', 
      56: 'diapers wipes', 85: 'food storage', 117: 'nuts seeds dried fruit', 25: 'soap', 
      123: 'packaged vegetables fruits', 106: 'hot dogs bacon sausage', 96: 'lunch meat', 
      107: 'chips pretzels', 122: 'meat counter', 67: 'fresh dips tapenades', 1: 'prepared soups salads', 
      72: 'condiments', 98: 'juice nectars', 99: 'canned fruit applesauce', 51: 'preserved dips spreads', 
      32: 'packaged produce', 81: 'canned jarred vegetables', 12: 'fresh pasta', 9: 'pasta sauce', 
      116: 'frozen produce', 129: 'frozen appetizers sides', 69: 'soup broth bouillon', 131: 'dry pasta', 
      13: 'prepared meals', 16: 'fresh herbs', 130: 'hot cereal pancake mixes', 104: 'spices seasonings', 
      63: 'grains rice dried goods', 58: 'frozen breads doughs', 100: 'missing', 23: 'popcorn jerky', 
      57: 'granola', 133: 'muscles joints pain relief', 64: 'energy sports drinks', 78: 'crackers', 
      45: 'candy chocolate', 50: 'fruit vegetable snacks', 128: 'tortillas flat bread', 14: 'tofu meat alternatives', 
      27: 'beers coolers', 75: 'laundry', 66: 'asian foods', 34: 'frozen meat seafood', 38: 'frozen meals', 
      88: 'spreads', 46: 'mint gum', 11: 'cold flu allergy', 93: 'breakfast bakery', 125: 'trail mix snack mix', 
      101: 'air fresheners candles', 126: 'feminine care', 48: 'breakfast bars pastries', 4: 'instant foods', 
      124: 'spirits', 89: 'salad dressing toppings', 105: 'doughs gelatins bake mixes', 19: 'oils vinegars', 
      92: 'baby food formula', 44: 'eye ear care', 40: 'dog food care', 82: 'baby accessories', 79: 'frozen pizza', 
      5: 'marinades meat preparation', 42: 'frozen vegan vegetarian', 55: 'shave needs', 134: 'specialty wines champagnes', 
      61: 'cookies cakes', 114: 'cleaning products', 15: 'packaged seafood', 68: 'bulk grains rice dried goods', 
      119: 'frozen dessert', 109: 'skin care', 80: 'deodorants', 62: 'white wines', 65: 'protein meal replacements', 
      95: 'canned meat seafood', 70: 'digestion', 60: 'trash bags liners', 71: 'refrigerated pudding desserts', 
      2: 'specialty cheeses', 18: 'bulk dried fruits vegetables', 28: 'red wines', 127: 'body lotions soap', 
      22: 'hair care', 47: 'vitamins supplements', 90: 'cocoa drink mixes', 118: 'first aid', 74: 'dish detergents', 
      7: 'packaged meat', 6: 'other', 41: 'cat food care', 76: 'indian foods', 97: 'baking supplies decor', 
      39: 'seafood counter', 103: 'ice cream toppings', 102: 'baby bath body care', 87: 'more household', 
      33: 'kosher foods', 73: 'facial care', 10: 'kitchen supplies', 132: 'beauty', 113: 'frozen juice'
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

      const response = await fetch('http://127.0.0.1:5001/recommend', {
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
      <h1>Product Recommendation</h1>

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

      <style jsx>{`
        .container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 20px;
          font-family: Arial, sans-serif;
          text-align: center;
        }

        h1 {
          color: #4CAF50;
          margin-bottom: 20px;
          font-size: 28px;
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
          color: black; /* Ensures text is visible */
          width: 100%;
          box-sizing: border-box; /* Prevent overflow */
        }

        select:focus {
          outline: 2px solid #4CAF50;
        }

        .button {
          padding: 12px 20px;
          font-size: 18px;
          background-color: #4CAF50;
          color: white;
          border: none;
          border-radius: 5px;
          cursor: pointer;
          margin-top: 20px;
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

        .response li {
          margin: 5px 0;
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
