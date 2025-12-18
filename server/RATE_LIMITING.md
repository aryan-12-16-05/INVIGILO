# Rate Limiting Configuration

## Overview
Rate limiting has been implemented to protect the INVIGILO API from abuse, brute force attacks, and excessive usage.

## Implementation Details

### Library Used
- **Package:** `flask-limiter` v3.5.0
- **Storage:** In-memory (can be upgraded to Redis for distributed systems)
- **Key Function:** `get_remote_address()` - tracks requests by client IP address

### Global Configuration

**Default Limits (applies to all endpoints):**
- 1000 requests per day per IP
- 200 requests per hour per IP

**Storage:**
- Currently using in-memory storage (`memory://`)
- For production with multiple servers, upgrade to Redis: `redis://localhost:6379`

### Protected Endpoints

| Endpoint | Rate Limit | Reason |
|----------|------------|--------|
| `POST /api/register` | 5 per hour | Prevent spam account creation |
| `POST /api/login` | 10 per hour | Prevent brute force password attacks |
| `POST /api/verify-face` | 20 per hour | Prevent face matching attacks |
| `POST /api/proctor` | 100 per hour | High volume during exam, but prevent abuse |

### Error Response

When rate limit is exceeded, the API returns:

**Status Code:** `429 Too Many Requests`

**Response Body:**
```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Please try again later.",
  "retry_after": 3600
}
```

**Headers:**
- `Retry-After`: Time in seconds until rate limit resets

## Installation

### 1. Install Package
```powershell
cd server
pip install flask-limiter==3.5.0
```

Or install all dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Verify Installation
```powershell
python -c "from flask_limiter import Limiter; print('✓ flask-limiter installed')"
```

## Usage Examples

### Testing Rate Limits

**Test Registration Limit (5/hour):**
```powershell
# Make 6 rapid requests to trigger rate limit
for ($i=1; $i -le 6; $i++) {
    curl -X POST http://localhost:5000/api/register `
         -H "Content-Type: application/json" `
         -d '{"email":"test@example.com","password":"test123","role":"student"}'
    Write-Host "Request $i completed"
}
```

**Expected Result:**
- First 5 requests: Normal response (200/400/500 depending on validity)
- 6th request: `429 Too Many Requests` with retry-after header

**Test Login Limit (10/hour):**
```powershell
# Make 11 rapid requests
for ($i=1; $i -le 11; $i++) {
    curl -X POST http://localhost:5000/api/login `
         -H "Content-Type: application/json" `
         -d '{"identifier":"test@example.com","password":"wrong","role":"student"}'
    Write-Host "Request $i completed"
}
```

**Expected Result:**
- First 10 requests: Normal response
- 11th request: `429 Too Many Requests`

### Bypassing During Development

If you need to disable rate limiting temporarily for testing:

**Method 1: Exempt specific IP (localhost)**
```python
# In app.py, add after limiter initialization:
limiter.exempt("127.0.0.1")
```

**Method 2: Disable globally**
```python
# In app.py, modify limiter initialization:
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    enabled=False  # Disable rate limiting
)
```

**Method 3: Increase limits for development**
```python
# Temporarily increase limits
@app.route('/api/register', methods=['POST'])
@limiter.limit("1000 per hour")  # Much higher limit
def register_user():
    # ...
```

## Monitoring

### Log Rate Limit Violations

All rate limit violations are automatically logged:

```python
app.logger.warning(f"Rate limit exceeded from {get_remote_address()}: {e.description}")
```

**Log Output Example:**
```
[2025-12-14 10:23:45] WARNING: Rate limit exceeded from 192.168.1.100: 5 per 1 hour
```

### View Rate Limit Stats

To see current rate limit status for an IP:

```python
# Add this debug endpoint (remove in production)
@app.route('/api/rate-limit-status', methods=['GET'])
def rate_limit_status():
    from flask_limiter.util import get_remote_address
    ip = get_remote_address()
    # Get limit info (flask-limiter internal)
    return jsonify({"ip": ip, "message": "Check server logs for details"})
```

## Customization

### Adjust Rate Limits

Edit the decorator values in `app.py`:

**More Strict (Increase Security):**
```python
@limiter.limit("3 per hour")  # Only 3 registration attempts
def register_user():
    # ...

@limiter.limit("5 per hour")  # Only 5 login attempts
def login_user():
    # ...
```

**More Lenient (Better User Experience):**
```python
@limiter.limit("10 per hour")  # 10 registration attempts
def register_user():
    # ...

@limiter.limit("20 per hour")  # 20 login attempts
def login_user():
    # ...
```

### Per-User Rate Limiting

Instead of IP-based, you can limit by user ID:

```python
def get_user_id():
    # Extract user ID from request headers or JWT token
    return request.headers.get('X-User-Id', get_remote_address())

limiter = Limiter(
    app=app,
    key_func=get_user_id,  # Use user ID instead of IP
    # ...
)
```

### Different Limits for Different Roles

```python
@app.route('/api/proctor', methods=['POST'])
def proctor_activity():
    user = get_current_user()  # Your auth function
    
    # Apply different limits based on role
    if user.role == 'student':
        limiter.limit("100 per hour")(lambda: None)
    else:  # lecturer
        limiter.limit("500 per hour")(lambda: None)
    
    # ... rest of function
```

## Production Upgrade

### Use Redis for Distributed Systems

When running multiple Flask instances (load balancing), upgrade to Redis:

**1. Install Redis:**
```powershell
# Windows: Download from https://redis.io/download
# Or use Docker:
docker run -d -p 6379:6379 redis:latest
```

**2. Install Redis Client:**
```powershell
pip install redis
```

**3. Update app.py:**
```python
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379",  # Use Redis instead of memory
)
```

**Benefits:**
- ✅ Shared rate limits across multiple servers
- ✅ Persistent rate limit state (survives server restarts)
- ✅ Better performance for high-traffic scenarios

### Environment-Based Configuration

Use environment variables for production:

```python
import os

RATE_LIMIT_STORAGE = os.getenv('RATE_LIMIT_STORAGE', 'memory://')
RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=RATE_LIMIT_STORAGE,
    enabled=RATE_LIMIT_ENABLED,
)
```

**.env file:**
```env
# Development
RATE_LIMIT_STORAGE=memory://
RATE_LIMIT_ENABLED=false

# Production
RATE_LIMIT_STORAGE=redis://redis-server:6379
RATE_LIMIT_ENABLED=true
```

## Troubleshooting

### Issue: Rate limit triggered too quickly

**Solution:** Increase the limit or time window
```python
@limiter.limit("10 per hour")  # Change to "20 per hour"
```

### Issue: Legitimate users getting blocked

**Possible Causes:**
1. Multiple users behind same IP (NAT/proxy)
2. Rate limit too strict
3. Automated testing triggering limits

**Solutions:**
1. Switch to user-based rate limiting (requires authentication)
2. Increase limits for specific routes
3. Whitelist testing IPs

### Issue: Rate limiting not working

**Check:**
1. flask-limiter installed: `pip show flask-limiter`
2. Limiter initialized after `app = Flask(__name__)`
3. Decorator placed ABOVE `@app.route()`, not below
4. Check if exempted: `limiter.exempt()` was not called

### Issue: 429 error during legitimate exam

**Scenario:** Student taking long exam hits 100/hour limit on `/api/proctor`

**Solution:** Increase limit for proctoring endpoint
```python
@limiter.limit("300 per hour")  # Allow 1 request every 12 seconds for 1 hour
def proctor_activity():
    # ...
```

## Security Best Practices

### 1. Combine with Other Security Measures

Rate limiting is one layer of defense. Also implement:
- ✅ Strong password requirements
- ✅ CAPTCHA for registration/login after failed attempts
- ✅ Account lockout after N failed login attempts
- ✅ IP blacklisting for persistent attackers
- ✅ JWT token expiration
- ✅ HTTPS/TLS encryption

### 2. Monitor and Alert

Set up monitoring for:
- High rate of 429 responses (possible attack)
- Specific IPs hitting limits frequently
- Unusual traffic patterns

### 3. Dynamic Rate Limiting

Adjust limits based on detected threats:
```python
# Pseudo-code
if detect_brute_force_attack(ip):
    limiter.limit("1 per hour")  # Dramatically reduce limit
else:
    limiter.limit("10 per hour")  # Normal limit
```

## Summary

✅ **Implemented:** Rate limiting on 4 critical endpoints  
✅ **Security:** Prevents brute force and spam attacks  
✅ **Scalability:** Can upgrade to Redis for distributed systems  
✅ **Monitoring:** All violations logged for security analysis  
✅ **Customizable:** Easy to adjust limits per endpoint  
✅ **User-Friendly:** Clear error messages with retry information  

**Files Modified:**
- `server/requirements.txt` - Added flask-limiter==3.5.0
- `server/app.py` - Added limiter initialization and decorators
- `server/RATE_LIMITING.md` - This documentation

**Next Steps:**
1. Install flask-limiter: `pip install flask-limiter==3.5.0`
2. Test rate limits with rapid requests
3. Monitor 429 responses in production
4. Adjust limits based on real usage patterns
5. Consider upgrading to Redis for production deployment
