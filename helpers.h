
const ll MOD = 1000000007;

const char alphanum [] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "abcdefghijklmnopqrstuvwxyz";

template <typename T>
T mod(const T& a, const T& m)
{
  if (a>=0) return a%m;
  T k = (-a)/m;
  return (a+(k+1)*m)%m;
}

template <typename T>
T s_mod(const T& a) {assert(a>=0); return a%MOD;}

template <typename T>
T mod(const T& a) {return a%MOD;}

// to be used with c++11 ranged-base loop
vii neigh(int i, int j, int n, int m)
{
  vii v;
  int di [] = {1, 0, -1, 0};
  int dj [] = {0, 1, 0, -1};
  int ni, nj;
  int k;
  FOR(k, 4) {
    ni=i+di[k];
    nj=j+dj[k];
    if (OUT(ni, n) || OUT(nj, m)) continue;
    v.pb({ni, nj}); // c++11 list_initializer
  }
  return v;
}

// leverage c++11 ranged-base loop and lambda functions...
template <typename Proc>
void proc_neigh(int i, int j, int n, int m, Proc proc)
{
  for (ii& nn : neigh(i, j, n, m)) proc(nn.fi, nn.se);
}

template <typename InputIterator>
InputIterator::value_type sum(InputIterator first, InputIterator last)
{
  typedef InputIterator::value_type T;
  T s = 0;
  for_each(first, last, [&s] (T i) {s+=i;});
  return s;
}

ll gcd(ll u, ll v)
{
  while ( v != 0) {
    ll r = u % v;
    u = v;
    v = r;
  }
  return u;
}

bool isPrime(ll number) {
  if (number<2) return false;
  if (number==2) return true;
  ll i;
  for (i=2; i*i<=number; i++)
  {
    if (number%i == 0)
    {
      return false;
    }
  }
  return true;
}

template <typename T, int N>
void print(T (&vec) [N])
{
  for (int i = 0; i<N; i++) {
    cout << vec[i] << " ";
  }
  cout << endl;
}

template <typename T, int N>
void print(T (&vec) [N], int n)
{
  for (int i = 0; i<n; i++) {
    cout << vec[i] << " ";
  }
  cout << endl;
}

template <typename T>
void print(vector<T> v)
{
  for (int i = 0; i<v.size(); i++) {
    cout << v[i] << " ";
  }
  cout << endl;
}

void print(vs v)
{
  for (int i = 0; i<v.size(); i++) {
    cout << v[i] << endl;
  }
  cout << endl;
}

template <typename T>
void print(initializer_list<T> il)
{
  vector<T> v(ALL(il));
  for (int i = 0; i<v.size(); i++) {
    cout << v[i] << " ";
  }
  cout << endl;
}

template <typename T, int N, int M>
void print(T (&mat) [N][M])
{
  for (int i = 0; i<N; i++) {
    for (int j = 0; j<M; j++) {
      cout <<mat[i][j] << " ";
    }
    cout << endl;
  }
  cout << endl;
}

template <typename T, int M>
void print(T mat [][M], int n, int m)
{
  for (int i = 0; i<n; i++) {
    for (int j = 0; j<m; j++) {
      cout << mat[i][j] << " ";
    }
    cout << endl;
  }
  cout << endl;
}

// useless with c++11 lambda functions
template <typename T>
struct Comp
{
  Comp() {}
  bool operator() (const T& a, const T& b) {return a < b;}
};

typedef int result_type;
result_type dt [51];
result_type compute(int a)
{
  // corner cases

  // check dt
  result_type& res = dt[a];
  if (res > -1) return res;


  // compute res
  res=0;

  return res;
}

/*
 * compute1 : # n-digit numbers s.t. the sum of the digits is sum, and no leading zeros
 * compute2 : # n-digit numbers s.t. the sum of the digits is sum, with possible leading zeros
 */
ll dt1 [18][163];
ll dt2 [18][163];
ll compute1(int n, int sum)
{
  if (sum<0||sum>162) return 0;
  if (n == 0) {if (1<=sum && sum<10) return 1; else return 0;}
  ll& res = dt1[n][sum];
  if (res > -1) return res;
  res = 0;
  for (int d=0; d<10; d++) {
     res += compute1(n-1, sum-d);
  }
  return res;
}
ll compute2(int n, int sum)
{
  if (n==-1 && sum==0) return 1;
  if (sum<0||sum>162) return 0;
  if (n == 0) {if (0<=sum && sum<10) return 1; else return 0;}
  ll& res = dt2[n][sum];
  if (res > -1) return res;
  res = 0;
  for (int d=0; d<10; d++) {
     res += compute2(n-1, sum-d);
  }
  return res;
}

template <int P>
class Modint
{
ll a_;
public:
Modint(const ll& a=0) : a_(a%P) {}
operator ll() const {return a_;}
Modint& operator=(const ll& a) {a_ = a%P; return *this;}
Modint& operator+=(const Modint& a)
{
  a_ += a;
  a_ %= P;
  return *this;
}
Modint& operator-=(const Modint& a)
{
  a_ += P-a;
  a_ %= P;
  return *this;
}
Modint& operator*=(const Modint& a)
{
  a_ *= a;
  a_ %= P;
  return *this;
}
friend Modint operator+(const Modint& a, const Modint& b)
{
  Modint c = a;
  c += b;
  return c;
}
friend Modint operator-(const Modint& a, const Modint& b)
{
  Modint c = a;
  c -= b;
  return c;
}
friend Modint operator-(const Modint& a)
{
  Modint c;
  c -= a;
  return c;
}
friend Modint operator*(const Modint& a, const Modint& b)
{
  Modint c = a;
  c *= b;
  return c;
}
Modint pow(int p)
{
  ll a = a_;
  ll res = 1;
  while(p>0) {
    if (p%2==1) res *= a, res %= P;
    a *= a, a %= P, p/=2;
  }
  return res;
}
Modint inv()
{
  return this->pow(P-2);
}
};

typedef Modint<MOD> mint;

template <int P>
class Fac
{
typedef Modint<P> mint;
typedef vector<mint> vmint;
vmint v_;
public:
Fac(int n)
{
  v_ = vmint(n+1, 0);
  int i;
  mint acc = 1;
  v_[0] = acc;
  FOR2(i, 1, n+1) acc *= i, v_[i] = acc;
}
mint operator()(int i)
{
  return v_[i];
}
};

template <int P>
class InvFac
{
typedef Modint<P> mint;
typedef vector<mint> vmint;
vmint v_;
public:
InvFac(int n)
{
  v_ = vmint(n+1, 0);
  int i;
  mint acc = 1;
  v_[0] = acc;
  FOR2(i, 1, n+1) acc *= i, v_[i] = acc.inv();
}
mint operator()(int i)
{
  return v_[i];
}
};

template <int P>
class Fac2
{
typedef Modint<P> mint;
typedef vector<mint> vmint;
ll nmin, nmax;
vector<vmint> v_;
public:
Fac2(ll a, ll b)
: nmin(a), nmax(b) 
{
  v_ = vector<vmint>(b-a+1, vmint(b-a+1, 0));
  ll i;
  FOR(i, b-a+1) v_[i][i]=1;
  FOR(i, b-a+1) {
    mint acc = a+i;
    ll j;
    FOR2(j, i+1, b-a+1) v_[i][j] = acc, acc *= a+j;
  }
}
mint operator()(ll i, ll j)
{ 
  return v_[i-nmin][j-nmin];
}
};

template <int P>
class Choose
{
typedef Modint<P> mint;
typedef vector<mint> vmint;
ll nmin, nmax, kmax;
vector<vmint> v_;
public:
Choose(ll nmin, ll nmax, ll kmax)
: nmin(nmin), nmax(nmax), kmax(kmax)
{
  InvFac<P> invf(kmax+1);
  Fac2<P> f2(max(0LL, nmin-kmax), nmax+1);

  v_ = vector<vmint>(nmax-nmin+1, vmint(kmax+1, 0));
  ll i, j;
  FOR(i, nmax-nmin+1) FOR(j, kmax+1) if (j <= i+1+nmin) v_[i][j] = f2(i-j+1+nmin,i+1+nmin)*invf(j);
}
mint operator()(ll n, ll k)
{
  return v_[n-nmin][k];
}
};

// kmp table
vector<vi> kmp_table(string s)
{
  vector<vi> v;
  if (s.empty()) return v;
  vi tmp(256, 0);
  tmp[s[0]]=1;
  v.pb(tmp);
  int i, j, k;
  j=0;
  FOR2(i, 1, s.size()) {
    fill(ALL(tmp), 0);
    FOR(k, 256) {
      if (k==s[i]) tmp[k]=i+1;
      else tmp[k]=v[j][k];
    }
    j=v[j][s[i]];
    v.pb(tmp);
  }
  return v;
}

// square matrix multiplication
template <typename T, int N>
void mamul(T (&A) [N][N], T (&B) [N][N], T (&C) [N][N], int n)
{
  T tmp [N][N];
  ZERO(T);
  int i, j, k;
  FOR(i, n) FOR(j, n) FOR(k, n) tmp[i][j] = B[i][k] * C[k][j];
  FOR(i, n) FOR(j, n) A[i][j] = tmp[i][j];
}

// https://bosker.wordpress.com/2014/02/18/the-bicycle-lock-problem/
int bicycle_lock(string S, string T)
{
  int n=S.size();
  vi tmp(n), D (n);
  int i;
  FOR(i, n) tmp[i] = mod(S[i]-T[i], 10);
  tmp.insert(tmp.begin(), 0);
  FOR(i, n) D[i] = mod(tmp[i+1]-tmp[i], 10);
  int s =0;
  FOR(i, n) s+=D[i];
  int x = s/10;
  sort(ALL(D));
  int res=0;
  FOR(i, n-x) res+=D[i];
  return res;
}

// not tested
vi manacher_table(string s)
{
  int n = s.size();
  vi v(2*n-1);
  int c, r, i;
  r=c=i=0;
  FOR(i, 2*n-1) {
    int op = 2*c-i;
    if((i-1)/2+v[op])>=r) {
      c=i;
      while(r+1<n && c-r-1 >=0 && s[r+1]==s[c-r-1]) r++;
      v[c] = r-(c-1)/2;
    }
    else {
      v[i] = v[op];
    }
  }
  return v;
}

// not tested
// dep [i][j] = 1 if i depends on j
//            = 0 if j eliminated
//            = -1 does not depend on j
template <int M>
vi topological_order(int dep [][M])
{
  int i, j;
  vi res;
  bool stop = false
  while(!stop) {
    stop = true;
    FOR(i, M) {
      bool ok=false;
      FOR(j, M) if (dep[j][i] >=0) {ok = true; break;}
      if (!ok) {
        res.pb(i);
        stop = false;
        FOR(j, M) dep[j][i]=0;
        FOR(j, M) if (dep[i][j]>0) dep[i][j]=-1;
      }
    }
  }
  return res;
}

ll euler_totient(ll n)
{
  ll p=2;
  ll r = n;
  while(p*p <= n) {
    if (n%p==0) {
      r-=r/p;
      while (n%p==0) n/=p;
    }
    p++;
  }

  if (n>1) {
    r-=r/n;
  } 

  return r;
}

// is string a a subsequence of string b?
bool sub_sequence(string a, string b)
{
  int i = 0;
  int j = 0;
  while ( (i < b.size()) && (j < a.size() ) ) {
    if (a[j] == b[i]) {
      j ++;
    }
    i ++;
  }
  return (j == a.size() ); 
}

// map is an associative array: opos[ ')' ] returns '(', 
// opos[ ']' ] is '[', ...
bool correctBracket(string exp, map<char, char> opos)
{
  stack<char> s;
  for (char ch: exp) {
    // we push opening brackets to the stack
    if (opos.find(ch) != opos.end()) {
      s.push(ch);
    } else {
      // If we find a closing bracket, we make sure it matches the
      // opening bracket in the top of the stack
      if (s.size() == 0 || s.top() != opos[ch]) {
        return false;
      } else {
        // then we remove it
        s.pop();
      }
    }
  }
  // stack must be empty.
  return s.empty();
}

template <typename T, int N>
void floydwarshall(T (&dist) [N][N])
{
  int i, j, k;
  FOR(k, N) FOR(i, N) FOR(j, N) {
    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
  }
}

// compute the lex-smallest string with letters a-z from suffix array
string lexsmallestfromsuffixarray(vi suffixarray)
{
  int n = suffixarray.size();

  // rank[i] will provide the rank of suffix i
  vi rank (n+1);
  int i;

  FOR(i, n) rank[suffixarray[i]] = i;
  rank[n] = -1;

  string res (n, 'a');
  FOR2(i, 1, n) {
    res[suffixarray[i]] = res[suffixarray[i-1]] + 
        (rank[suffixarray[i]+1] < rank[suffixarray[i-1]+1]);
  }

  return res;
}

// segment tree

int segment_tree [4*MAX];

int query(int node, int left, int right, int queryleft, int queryright)
{
    // cover
    if (left>=queryleft && queryright>=right) return segment_tree[node];

    // empty
    if (left==right) return 0;

    // no intersection
    if (left>=queryright || right<=queryleft) return 0;

    // partial overlap
    int mid=(left+right)/2;
    return query(2*node+1, left, mid, queryleft, queryright) + query(2*node+2, mid, right, queryleft, queryright);
}

void update(int node, int left, int right, int idx, int val)
{
    // singleton
    if (left+1==right && idx==left) {
        segment_tree[node]=val;
        return;
    }

    // in interval
    if (idx>=left && idx<right) {
        int mid=(left+right)/2;
        update(2*node+1, left, mid, idx, val);
        update(2*node+2, mid, right, idx, val);
        segment_tree[node]=segment_tree[2*node+1]+segment_tree[2*node+2];
    }
}
