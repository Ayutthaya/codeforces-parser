// Create your own template by modifying this file!
#include <bits/stdc++.h>
using namespace std;

#define PI 3.1415926535897932384626433832795028841971693992

#define pb push_back
#define mp make_pair 
#define fi first
#define se second
#define ZERO(a) memset(a, 0, sizeof(a))
#define INIT(a) memset(a, 0xff, sizeof(a))
#define FOR(i, n) for (i=0; i<n; i++)
#define FORD(i, n) for (i=n-1; i>=0; i--)
#define FOR2(i, a, b) for (i=a; i<b; i++)
#define FOR2D(i, a, b) for (i=b-1; i>=a; i--)
#define BIT(i, m) (m&(1LL<<(i)))
#define RBIT(i, m) (m&~(1LL<<(i)))
#define ABIT(i, m) (m|(1LL<<(i)))
#define FORB(i, m) for (i=0; i<32; i++) if (BIT(i, m)>0)
#define ALL(a) a.begin(), a.end()
#define OUT(i, n) (i<0 || i>=n)
#define AT(a, i) (a[(i+a.size())%a.size()])
#define IT(v, i) (v.begin()+(i+v.size())%v.size())
#define IT2(v, i, j) IT(v, i), IT(v, j)
#define INSET(a, S) (S.find(a)!=S.end())

typedef vector<int> vi;
typedef long long ll;
typedef vector<ll> vll;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef vector<string> vs;
typedef list<int> li;
typedef long double ld;

#ifdef DEBUG
     #define debug(args...)            {dbg,args; cerr<<endl;}
#else
    #define debug(args...)              // Just strip off all debug tokens
#endif

struct debugger
{
    template<typename T> debugger& operator , (const T& v)
    {    
        cerr<<v<<" ";    
        return *this;    
    }
} dbg;

int main() 
{

}

