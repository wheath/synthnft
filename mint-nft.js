require('dotenv').config();
const API_URL = process.env.API_URL;
const { createAlchemyWeb3 } = require("@alch/alchemy-web3");
const web3 = createAlchemyWeb3(API_URL);

const contract = require("./TimH.json");
//console.log(JSON.stringify(contract.abi));


const contractAddress = "0xc094661df60a5571b033217702febaeec6d98b21";
const nftContract = new web3.eth.Contract(contract.abi, contractAddress);

const accountFrom = {
  privateKey: 'nyb',
  address: '0xbC15291A35a5B06CaD79002DCac38978E3C7C85D',
};

const addressTo = '0xbC15291A35a5B06CaD79002DCac38978E3C7C85D';
const uriMp3 = 'QmXpwKTMKNrm5rkavHQqS1ddcAA7xpqoaPzR6uFoPmuvZo';




// 3. Create send function
const send = async () => {
  console.log(`Attempting to send transaction from ${accountFrom.address} to ${addressTo}`);

  // 4. Sign tx with PK
  const createTransaction = await web3.eth.accounts.signTransaction(
    {
      gas: 210000,
      to: contractAddress,
      data: nftContract.methods.safeMint(addressTo, uriMp3).encodeABI(),
    },
    accountFrom.privateKey
  );

  // 5. Send tx and wait for receipt
  const createReceipt = await web3.eth.sendSignedTransaction(createTransaction.rawTransaction);
  console.log(`Transaction successful with hash: ${createReceipt.transactionHash}`);
};

// 6. Call send function
send();



