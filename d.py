# Create a TestClient instance for the FastAPI application
client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_database():
    """
    This fixture runs before every test to reset the in-memory database.
    'autouse=True' means it will be automatically applied to all tests.
    """
    init_db()  # Call a function to reset the database to its initial state
    yield

def test_initial_balance():
    """
    Test to verify the initial balance of the first account.
    """
    response = client.get("/balance/1")
    assert response.status_code == 200
    assert response.json() == {"account_id": 1, "balance": 1000.0}

def test_deposit_successful():
    """
    Test a successful deposit and verify the balance is updated.
    """
    response = client.post("/deposit/1", json={"amount": 500.0})
    assert response.status_code == 200
    assert response.json()["message"] == "Deposit successful."
    assert response.json()["new_balance"] == 1500.0

def test_deposit_negative_amount():
    """
    Test a deposit with a negative amount, which should be rejected.
    """
    response = client.post("/deposit/1", json={"amount": -100.0})
    assert response.status_code == 400
    assert "Cannot deposit a negative amount" in response.json()["detail"]

def test_withdraw_successful():
    """
    Test a successful withdrawal and verify the balance is updated.
    """
    response = client.post("/withdraw/1", json={"amount": 200.0})
    assert response.status_code == 200
    assert response.json()["message"] == "Withdrawal successful."
    assert response.json()["new_balance"] == 800.0

def test_withdraw_insufficient_funds():
    """
    Test a withdrawal that exceeds the account balance, which should be rejected.
    """
    response = client.post("/withdraw/1", json={"amount": 2000.0})
    assert response.status_code == 400
    assert "Insufficient funds" in response.json()["detail"]

def test_transfer_successful():
    """
    Test a successful transfer between two accounts.
    """
    response = client.post("/transfer/", json={"from_id": 1, "to_id": 2, "amount": 300.0})
    assert response.status_code == 200
    assert response.json()["message"] == "Transfer successful."
    
    # Verify the balance of the sender
    sender_balance = client.get("/balance/1")
    assert sender_balance.json()["balance"] == 700.0
    
    # Verify the balance of the receiver
    receiver_balance = client.get("/balance/2")
    assert receiver_balance.json()["balance"] == 1300.0

def test_transfer_insufficient_funds():
    """
    Test a transfer with insufficient funds in the source account.
    """
    response = client.post("/transfer/", json={"from_id": 1, "to_id": 2, "amount": 5000.0})
    assert response.status_code == 400
    assert "Insufficient funds in the source account" in response.json()["detail"]

def test_transfer_non_existent_account():
    """
    Test a transfer to an account that does not exist.
    """
    response = client.post("/transfer/", json={"from_id": 1, "to_id": 99, "amount": 100.0})
    assert response.status_code == 404
    assert "Account not found" in response.json()["detail"]
 
https://allianz-agn.webex.com/allianz-agn/j.php?MTID=m28730fd7526f86aeab390cf297020a50
