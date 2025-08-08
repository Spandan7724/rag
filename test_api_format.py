#!/usr/bin/env python3
"""
Quick API Format Test
Test the correct API format with a single document
"""
import asyncio
import aiohttp
import json

async def test_api_format():
    """Test the API with the correct format"""
    
    # Test data based on your curl example
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
        "questions": [
            "What is the freelook period for this policy?",
            "Is there a waiting period for pre-existing diseases?",
            "What is the initial waiting period for any illness claim (other than accidental)?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 8915ddf1d1760f2b6a3b027c6fa7b16d2d87a042c41452f49a1d43b3cfa6245b"
    }
    
    print("🧪 Testing API Format")
    print("🌐 URL: http://localhost:8000/hackrx/run")
    print(f"📝 Payload: {json.dumps(payload, indent=2)}")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://localhost:8000/hackrx/run",
                json=payload,
                headers=headers,
                timeout=300  # 5 minute timeout
            ) as response:
                print(f"📊 Status Code: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ Success!")
                    print(f"📄 Response type: {type(result)}")
                    
                    if isinstance(result, list):
                        print(f"📋 Number of responses: {len(result)}")
                        for i, item in enumerate(result):
                            print(f"  [{i+1}] Response type: {type(item)}")
                            if isinstance(item, dict):
                                print(f"      Keys: {list(item.keys())}")
                                answer = item.get("answer", "No answer key")
                                print(f"      Answer preview: {answer[:100]}...")
                    else:
                        print(f"📋 Single response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                        
                    # Save full response for inspection
                    with open("api_test_response.json", "w") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print("💾 Full response saved to api_test_response.json")
                    
                else:
                    error_text = await response.text()
                    print(f"❌ Error {response.status}: {error_text}")
                    
        except Exception as e:
            print(f"💥 Request failed: {str(e)}")

if __name__ == "__main__":
    print("🔧 Make sure your API server is running on http://localhost:8000")
    print("⏳ This will test the correct API format...")
    
    asyncio.run(test_api_format())