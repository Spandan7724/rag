uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

python run.py



  Testing with curl:
  # Upload local file
curl -X POST "http://localhost:8000/hackrx/run" \                                        20:13:29
                             -H "Content-Type: application/json" \
                             -H "Authorization: Bearer 8915ddf1d1760f2b6a3b027c6fa7b16d2d87a042c41452f49a1d43b3cfa6245b"
 \
                             -d '{
                           "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07
-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                           "questions": ["Test timing"]
                         }'

  # Process local file directly
curl -X POST "http://localhost:8000/hackrx/run" \                                    
                             -H "Content-Type: application/json" \
                             -H "Authorization: Bearer 8915ddf1d1760f2b6a3b027c6fa7b16d2d87a042c41452f49a1d43b3cfa6245b"
 \
                             -d '{
                           "documents": "file:////home/spandan/projects/bajaj/pdfs/Arogya%20Sanjeevani%20Policy%20-%20CI
N%20-%20U10200WB1906GOI001713%201.pdf",
                           "questions": ["Test timing"]
                         }'


# Basic usage
  python test_local_file.py "/path/to/document.pdf" "What is covered?" "What are the limits?"