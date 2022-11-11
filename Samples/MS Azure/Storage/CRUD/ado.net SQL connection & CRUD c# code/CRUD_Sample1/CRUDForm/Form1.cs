using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using System.Data.SqlClient;
using System.Collections;

namespace CRUDForm
{
    public partial class frmMain : Form
    {
        private const string CONNECTION_STRING = "Server=tcp:labuser26sqlserver.database.windows.net,1433;Initial Catalog=labuser26sql;Persist Security Info=False;User ID=fitersking;Password=rladudfks201!;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;";
        private SqlConnection SqlCon = null;
        private SqlCommand SqlCmd = null;
        private SqlDataAdapter SqlApt = new SqlDataAdapter();

        private DataSet dataMain = new DataSet();
        public frmMain()
        {
            InitializeComponent();
        }

        private void btnConnect_MouseClick(object sender, MouseEventArgs e)
        {
                
            btnConnect.Enabled = false;
        }

        private void btnGetData_Click(object sender, EventArgs e)
        {
            SqlCon = new SqlConnection(CONNECTION_STRING);
            string query = "SELECT * FROM production.brands";
            SqlCommand cmd = SqlCon.CreateCommand();
            cmd.CommandText = query;
            

            SqlApt.SelectCommand = cmd;
            SqlApt.Fill(dataMain);

            lstBrands.Items.Clear();

            DataRowCollection dataRows = dataMain.Tables[0].Rows;

            for (int i = 0;i< dataRows.Count;i++)
            {
                lstBrands.Items.Add(dataRows[i][1].ToString());
            }


            //fill to DataGridView
            DataSet DataProducts = new DataSet();
            query = "SELECT * FROM production.products";
            cmd.CommandText = query;
            SqlApt.Fill(DataProducts);
            grdProducts.DataSource = DataProducts.Tables[0];



            btnGetData.Enabled = false;
        }

        private void lstBrands_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (lstBrands.SelectedIndex == -1)
            {
                return;
            }
            
            
            //fill to DataGridView
            int selectedIndex = lstBrands.SelectedIndex;
            int selectedBrandID = Int32.Parse(dataMain.Tables[0].Rows[selectedIndex][0].ToString());


            DataSet DataProducts = new DataSet();
            string query = "SELECT * FROM production.products WHERE brand_id = @brand_id";
            SqlCommand cmd = SqlCon.CreateCommand();
            cmd.Parameters.Add(new SqlParameter("@brand_id", SqlDbType.Int)).Value=selectedBrandID;
            cmd.CommandText = query;
            SqlApt.SelectCommand = cmd;
            SqlApt.Fill(DataProducts);
            grdProducts.DataSource = DataProducts.Tables[0];
        }

        private void btnVIPmembers_MouseClick(object sender, MouseEventArgs e)
        {
            frmVIPMembers vip = new frmVIPMembers();
            vip.ShowDialog();
        }
    }
}
